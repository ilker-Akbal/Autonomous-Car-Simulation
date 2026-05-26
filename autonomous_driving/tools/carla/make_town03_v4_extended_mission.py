#!/usr/bin/env python3
import argparse
import json
import math
import os
import sys
from pathlib import Path


def add_carla_python_paths(carla_root: str):
    carla_root = os.path.expanduser(carla_root)
    paths = [
        os.path.join(carla_root, "PythonAPI", "carla"),
        os.path.join(carla_root, "PythonAPI"),
        os.path.expanduser("~/CARLA_DISK/PythonAPI/carla"),
        os.path.expanduser("~/İndirilenler/PythonAPI/carla"),
    ]

    for p in paths:
        if os.path.isdir(p) and p not in sys.path:
            sys.path.append(p)


def yaw_diff_deg(a, b):
    return abs((float(a) - float(b) + 180.0) % 360.0 - 180.0)


def feature_location(feature):
    props = feature.get("properties", {}) or {}
    geom = feature.get("geometry", {}) or {}
    coords = geom.get("coordinates", [0.0, 0.0])

    x = float(props.get("carla_x", coords[0]))
    y = float(props.get("carla_y", coords[1]))
    z = float(props.get("carla_z", 0.2))
    return x, y, z


def make_feature(carla, name, desc, nokta_id, wp, kind):
    loc = wp.transform.location
    rot = wp.transform.rotation

    return {
        "type": "Feature",
        "properties": {
            "name": name,
            "description": desc,
            "nokta_id": int(nokta_id),
            "yaw": 0.0,
            "town": "Town03",
            "carla_x": float(loc.x),
            "carla_y": float(loc.y),
            "carla_z": float(loc.z + 0.2),
            "carla_yaw": float(rot.yaw),
            "road_id": int(wp.road_id),
            "lane_id": int(wp.lane_id),
            "lane_width": float(wp.lane_width),
            "is_junction": bool(getattr(wp, "is_junction", False)),
            "kind": kind,
        },
        "geometry": {
            "type": "Point",
            "coordinates": [float(loc.x), float(loc.y)],
        },
    }


def nearest_driving_wp(carla, carla_map, x, y, z):
    loc = carla.Location(x=float(x), y=float(y), z=float(z))
    wp = carla_map.get_waypoint(
        loc,
        project_to_road=True,
        lane_type=carla.LaneType.Driving,
    )

    if wp is None:
        raise RuntimeError(f"Driving waypoint bulunamadı: x={x}, y={y}, z={z}")

    return wp


def choose_next_wp(current_wp, candidates):
    if not candidates:
        return None

    current_yaw = current_wp.transform.rotation.yaw

    def score(wp):
        dyaw = yaw_diff_deg(current_yaw, wp.transform.rotation.yaw)

        # Ters yöne dönmeyi ağır cezalandır.
        penalty = 0.0
        if dyaw > 100.0:
            penalty += 1000.0

        # Aynı lane/road devamını hafif tercih et.
        if int(wp.road_id) != int(current_wp.road_id):
            penalty += 5.0
        if int(wp.lane_id) != int(current_wp.lane_id):
            penalty += 3.0

        return dyaw + penalty

    return sorted(candidates, key=score)[0]


def follow_lane(carla, start_wp, extension_m, internal_step_m, via_spacing_m):
    current = start_wp
    travelled = 0.0
    last_added = 0.0
    out = []

    while travelled < extension_m:
        candidates = list(current.next(float(internal_step_m)))
        nxt = choose_next_wp(current, candidates)

        if nxt is None:
            print("[WARN] Yol devamı bulunamadı, uzatma burada kesildi.")
            break

        a = current.transform.location
        b = nxt.transform.location
        step_dist = math.hypot(float(b.x - a.x), float(b.y - a.y))

        travelled += step_dist
        current = nxt

        if travelled - last_added >= via_spacing_m:
            out.append((travelled, current))
            last_added = travelled

    return out, current, travelled


def draw_preview(carla, world, features, lifetime):
    color = carla.Color(0, 255, 80)

    prev = None
    for feature in features:
        props = feature.get("properties", {}) or {}
        x = float(props["carla_x"])
        y = float(props["carla_y"])
        z = float(props.get("carla_z", 0.2))
        loc = carla.Location(x=x, y=y, z=z + 0.4)

        if prev is not None:
            world.debug.draw_line(
                prev,
                loc,
                thickness=0.08,
                color=color,
                life_time=float(lifetime),
                persistent_lines=False,
            )

        if props.get("kind") in {"task", "park"} or props.get("name", "").startswith("via_final_"):
            world.debug.draw_string(
                loc + carla.Location(z=1.4),
                f'{props.get("nokta_id")} {props.get("name")}',
                draw_shadow=True,
                color=color,
                life_time=float(lifetime),
                persistent_lines=False,
            )

        prev = loc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--carla-root", default="/home/ilker/simulators/CARLA_0.9.15_SOURCE")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--timeout", type=float, default=30.0)

    parser.add_argument(
        "--base",
        default="autonomous_driving/missions/teknofest_town03_competition_v3_safe.geojson",
    )
    parser.add_argument(
        "--output",
        default="autonomous_driving/missions/teknofest_town03_competition_v4_extended.geojson",
    )

    parser.add_argument("--extension-m", type=float, default=420.0)
    parser.add_argument("--internal-step-m", type=float, default=5.0)
    parser.add_argument("--via-spacing-m", type=float, default=35.0)
    parser.add_argument("--draw-preview", action="store_true")
    parser.add_argument("--debug-lifetime", type=float, default=600.0)

    args = parser.parse_args()

    add_carla_python_paths(args.carla_root)
    import carla

    client = carla.Client(args.host, args.port)
    client.set_timeout(args.timeout)
    world = client.get_world()
    carla_map = world.get_map()

    print("CARLA map:", carla_map.name)

    base_path = Path(args.base).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    data = json.loads(base_path.read_text(encoding="utf-8"))
    features = data.get("features", [])

    if not features:
        raise RuntimeError("Base mission boş.")

    old_park = None
    copied = []

    for feature in features:
        props = feature.get("properties", {}) or {}
        name = props.get("name", "")

        new_feature = json.loads(json.dumps(feature))

        if name == "park_giris":
            old_park = json.loads(json.dumps(feature))
            new_props = new_feature["properties"]
            new_props["name"] = "via_eski_park_gecisi"
            new_props["description"] = "Eski park noktası; v4 extended rotada park öncesi geçiş noktası"
            new_props["kind"] = "via"
            copied.append(new_feature)
        else:
            copied.append(new_feature)

    if old_park is None:
        raise RuntimeError("Base mission içinde park_giris bulunamadı.")

    # Eski park noktasından devam edip final uzatma üret.
    x, y, z = feature_location(old_park)
    start_wp = nearest_driving_wp(carla, carla_map, x, y, z)

    extension_points, final_wp, travelled = follow_lane(
        carla=carla,
        start_wp=start_wp,
        extension_m=args.extension_m,
        internal_step_m=args.internal_step_m,
        via_spacing_m=args.via_spacing_m,
    )

    # Nokta sırasını yeniden yaz.
    out_features = []
    for feature in copied:
        feature["properties"]["nokta_id"] = len(out_features)
        out_features.append(feature)

    for idx, (d, wp) in enumerate(extension_points, start=1):
        out_features.append(
            make_feature(
                carla=carla,
                name=f"via_final_uzatma_{idx:02d}",
                desc=f"Park öncesi final uzatma geçiş noktası ({d:.1f} m)",
                nokta_id=len(out_features),
                wp=wp,
                kind="via",
            )
        )

    out_features.append(
        make_feature(
            carla=carla,
            name="park_giris",
            desc=f"V4 extended final park noktası; eski park sonrası yaklaşık {travelled:.1f} m uzatma",
            nokta_id=len(out_features),
            wp=final_wp,
            kind="park",
        )
    )

    out = {
        "type": "FeatureCollection",
        "features": out_features,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print("")
    print("YAZILDI:", output_path)
    print("Base:", base_path)
    print("Feature count:", len(out_features))
    print(f"Eklenen uzatma: requested={args.extension_m:.1f}m actual={travelled:.1f}m")
    print("")
    print("Son 15 nokta:")
    for feature in out_features[-15:]:
        p = feature["properties"]
        print(
            f'{p["nokta_id"]:02d} {p["name"]:24s} {p["kind"]:5s} '
            f'x={p["carla_x"]:.2f} y={p["carla_y"]:.2f} '
            f'road={p.get("road_id")} lane={p.get("lane_id")}'
        )

    if args.draw_preview:
        draw_preview(carla, world, out_features, args.debug_lifetime)
        print("CARLA üzerinde v4 rota preview çizildi.")


if __name__ == "__main__":
    main()
