#!/usr/bin/env python3
import argparse
import glob
import json
import math
import os
import sys
from pathlib import Path


def add_carla_paths(carla_root: str):
    carla_root = Path(os.path.expanduser(carla_root)).resolve()

    egg_pattern = str(
        carla_root
        / "PythonAPI"
        / "carla"
        / "dist"
        / f"carla-*{sys.version_info.major}.{sys.version_info.minor}-linux-x86_64.egg"
    )

    for egg in glob.glob(egg_pattern):
        if egg not in sys.path:
            sys.path.append(egg)

    for p in [
        carla_root / "PythonAPI" / "carla",
        carla_root / "PythonAPI",
    ]:
        if p.exists() and str(p) not in sys.path:
            sys.path.append(str(p))

    import carla
    return carla


def make_global_route_planner(carla_root: str, carla_map, resolution_m: float):
    add_carla_paths(carla_root)

    from agents.navigation.global_route_planner import GlobalRoutePlanner

    try:
        return GlobalRoutePlanner(carla_map, resolution_m)
    except TypeError:
        from agents.navigation.global_route_planner_dao import GlobalRoutePlannerDAO

        dao = GlobalRoutePlannerDAO(carla_map, resolution_m)
        planner = GlobalRoutePlanner(dao)
        planner.setup()
        return planner


def load_geojson(path: str):
    with open(os.path.expanduser(path), "r", encoding="utf-8") as f:
        data = json.load(f)

    if data.get("type") != "FeatureCollection":
        raise RuntimeError(f"{path}: GeoJSON kökü FeatureCollection olmalı.")

    return data


def feature_props(feature):
    return feature.get("properties", {}) or {}


def feature_name(feature):
    return str(feature_props(feature).get("name", "")).strip()


def feature_kind(feature):
    props = feature_props(feature)
    kind = str(props.get("kind", "")).strip().lower()
    name = feature_name(feature).lower()

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

    return "unknown"


def nokta_id(feature):
    try:
        return int(feature_props(feature).get("nokta_id", 999999))
    except Exception:
        return 999999


def point_xyz(feature):
    props = feature_props(feature)
    geom = feature.get("geometry", {}) or {}
    coords = geom.get("coordinates", []) or []

    x = props.get("carla_x", coords[0] if len(coords) > 0 else None)
    y = props.get("carla_y", coords[1] if len(coords) > 1 else None)
    z = props.get("carla_z", 0.2)

    if x is None or y is None:
        raise RuntimeError(f"Nokta x/y okunamadı: {feature_name(feature)}")

    return float(x), float(y), float(z)


def validate_tasks_only(tasks_data):
    names = [feature_name(f) for f in tasks_data.get("features", [])]
    via = [n for n in names if str(n).lower().startswith("via_")]

    if via:
        raise RuntimeError(f"tasks_only içinde via_* olmamalı: {via}")

    for required in ["start", "park_giris"]:
        if required not in names:
            raise RuntimeError(f"tasks_only içinde {required} yok.")

    tasks = [f for f in tasks_data.get("features", []) if feature_kind(f) == "task"]
    if not tasks:
        raise RuntimeError("tasks_only içinde görev noktası yok.")

    print("[OK] tasks_only GeoJSON yarışma formatında.")
    print("[OK] Rota/via noktası yok, sadece görev noktaları var.")


def mission_order(tasks_data):
    features = list(tasks_data.get("features", []))

    start = [f for f in features if feature_name(f).lower() == "start"]
    tasks = sorted([f for f in features if feature_kind(f) == "task"], key=nokta_id)
    park = [f for f in features if feature_name(f).lower() == "park_giris" or feature_kind(f) == "park"]

    if len(start) != 1:
        raise RuntimeError(f"start sayısı 1 olmalı, bulundu: {len(start)}")
    if len(park) != 1:
        raise RuntimeError(f"park_giris sayısı 1 olmalı, bulundu: {len(park)}")

    return start + tasks + park


def trace_full_route(planner, carla, ordered_features):
    full_route = []

    for i in range(len(ordered_features) - 1):
        a = ordered_features[i]
        b = ordered_features[i + 1]

        ax, ay, az = point_xyz(a)
        bx, by, bz = point_xyz(b)

        start_loc = carla.Location(x=ax, y=ay, z=az)
        end_loc = carla.Location(x=bx, y=by, z=bz)

        segment = planner.trace_route(start_loc, end_loc)

        if not segment:
            raise RuntimeError(f"Rota üretilemedi: {feature_name(a)} -> {feature_name(b)}")

        if full_route:
            segment = segment[1:]

        full_route.extend(segment)

        print(f"[OK] Segment: {feature_name(a)} -> {feature_name(b)} sample={len(segment)}")

    return full_route


def route_to_samples(route):
    samples = []
    total = 0.0
    prev = None

    for idx, item in enumerate(route):
        wp, road_option = item
        loc = wp.transform.location

        if prev is not None:
            total += math.hypot(loc.x - prev.x, loc.y - prev.y)

        option_name = getattr(road_option, "name", str(road_option)).upper()

        samples.append({
            "idx": idx,
            "x": float(loc.x),
            "y": float(loc.y),
            "z": float(loc.z),
            "yaw": float(wp.transform.rotation.yaw),
            "road_id": int(wp.road_id),
            "lane_id": int(wp.lane_id),
            "distance_m": float(total),
            "road_option": option_name,
        })

        prev = loc

    return samples


def min_distance_to_route(x, y, samples):
    best = None

    for s in samples:
        d = math.hypot(float(x) - s["x"], float(y) - s["y"])
        if best is None or d < best[0]:
            best = (d, s)

    return best


def compare_critical_reference(reference_data, samples):
    critical_names = [
        "via_rampa_giris",
        "via_tunel_giris",
        "via_tunel_ici",
        "via_tunel_cikis",
        "via_tunel_sonrasi",
        "gorev_1",
        "gorev_2",
        "via_alt_yol_1",
        "via_sag_dikey_orta",
        "via_ada_yaklasim",
        "via_final_uzatma_05",
        "gorev_3",
        "park_giris",
    ]

    by_name = {
        feature_name(f): f
        for f in reference_data.get("features", [])
        if feature_name(f)
    }

    print("\n[KRİTİK GÜZERGÂH KARŞILAŞTIRMASI]")
    warn_count = 0

    for name in critical_names:
        f = by_name.get(name)
        if f is None:
            print(f"[WARN] Referansta yok: {name}")
            warn_count += 1
            continue

        x, y, _ = point_xyz(f)
        best = min_distance_to_route(x, y, samples)

        if best is None:
            print(f"[FAIL] Route boş: {name}")
            warn_count += 1
            continue

        d, s = best
        status = "OK" if d <= 12.0 else "WARN"
        if status != "OK":
            warn_count += 1

        print(
            f"[{status}] {name:22s} "
            f"min_dist={d:6.2f} m "
            f"route_s={s['distance_m']:7.1f} m "
            f"road={s['road_id']:5d} lane={s['lane_id']:3d}"
        )

    print(f"\n[ÖZET] Kritik güzergâh WARN sayısı: {warn_count}")
    return warn_count


def save_route_geojson(samples, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    features = []
    for s in samples:
        features.append({
            "type": "Feature",
            "properties": {
                "idx": s["idx"],
                "distance_m": round(s["distance_m"], 3),
                "road_id": s["road_id"],
                "lane_id": s["lane_id"],
                "yaw": round(s["yaw"], 3),
                "road_option": s["road_option"],
            },
            "geometry": {
                "type": "Point",
                "coordinates": [s["x"], s["y"]]
            }
        })

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"type": "FeatureCollection", "features": features}, f, ensure_ascii=False, indent=2)

    print(f"[OK] Planner route GeoJSON yazıldı: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--carla-root", default="/home/ilker/simulators/CARLA_0.9.15")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--resolution", type=float, default=2.0)
    parser.add_argument(
        "--tasks",
        default="autonomous_driving/missions/teknofest_town03_competition_v4_tasks_only.geojson",
    )
    parser.add_argument(
        "--reference",
        default="autonomous_driving/missions/teknofest_town03_competition_v4_extended.geojson",
    )
    parser.add_argument(
        "--out",
        default="autonomous_driving/outputs/town03_map_reference/tasks_only_planned_route.geojson",
    )
    args = parser.parse_args()

    tasks_data = load_geojson(args.tasks)
    reference_data = load_geojson(args.reference)

    validate_tasks_only(tasks_data)
    ordered = mission_order(tasks_data)

    print("\n[GÖREV SIRASI]")
    for i, f in enumerate(ordered):
        x, y, z = point_xyz(f)
        print(f"{i:02d} {feature_name(f):12s} kind={feature_kind(f):6s} x={x:9.3f} y={y:9.3f} z={z:5.2f}")

    carla = add_carla_paths(args.carla_root)
    client = carla.Client(args.host, args.port)
    client.set_timeout(args.timeout)

    world = client.get_world()
    carla_map = world.get_map()

    print(f"\n[OK] CARLA bağlantısı var. Map: {carla_map.name}")

    planner = make_global_route_planner(args.carla_root, carla_map, args.resolution)
    route = trace_full_route(planner, carla, ordered)
    samples = route_to_samples(route)

    print(f"\n[OK] Runtime planner route üretti: sample={len(samples)} length={samples[-1]['distance_m']:.1f} m")

    compare_critical_reference(reference_data, samples)
    save_route_geojson(samples, args.out)

    print("\n[SONUÇ]")
    print("Bu testte GeoJSON rota vermedi. Rota, görev noktalarından CARLA GlobalRoutePlanner ile üretildi.")
    print("WARN çok çıkarsa görev noktaları aynı kalsa bile planner eski tabelalı güzergâhtan sapıyor demektir.")


if __name__ == "__main__":
    main()
