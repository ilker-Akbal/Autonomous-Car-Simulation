#!/usr/bin/env python3
import argparse
import json
import os
import sys


def add_carla_python_paths(carla_root: str):
    carla_root = os.path.expanduser(carla_root)
    paths = [
        os.path.join(carla_root, "PythonAPI", "carla"),
        os.path.join(carla_root, "PythonAPI"),
        os.path.expanduser("~/CARLA_DISK/PythonAPI/carla"),
        os.path.expanduser("~/İndirilenler/PythonAPI/carla"),
    ]

    for path in paths:
        if os.path.isdir(path) and path not in sys.path:
            sys.path.append(path)


def pick_location(world, carla):
    spec_tf = world.get_spectator().get_transform()
    start = spec_tf.location
    fwd = spec_tf.get_forward_vector()

    end = carla.Location(
        x=start.x + fwd.x * 300.0,
        y=start.y + fwd.y * 300.0,
        z=start.z + fwd.z * 300.0,
    )

    try:
        hits = world.cast_ray(start, end)
        if hits:
            hit = hits[0]
            return hit.location, "raycast"
    except Exception:
        pass

    return start, "spectator_location"


def make_feature(carla, world_map, name, nokta_id, loc, pick_mode):
    wp = world_map.get_waypoint(
        loc,
        project_to_road=True,
        lane_type=carla.LaneType.Driving,
    )

    if wp is None:
        raise RuntimeError("Yakın driving waypoint bulunamadı. Kamerayı yol/şerit üstüne baktır.")

    tf = wp.transform
    wloc = tf.location
    rot = tf.rotation

    props = {
        "name": name,
        "description": "Tünel/alt geçit rota ara noktası",
        "nokta_id": int(nokta_id),
        "kind": "via",
        "yaw": 0.0,
        "town": "Town03",
        "carla_x": float(wloc.x),
        "carla_y": float(wloc.y),
        "carla_z": float(wloc.z + 0.2),
        "carla_yaw": float(rot.yaw),
        "road_id": int(wp.road_id),
        "lane_id": int(wp.lane_id),
        "lane_width": float(wp.lane_width),
        "is_junction": bool(getattr(wp, "is_junction", False)),
        "pick_mode": pick_mode,
    }

    return {
        "type": "Feature",
        "properties": props,
        "geometry": {
            "type": "Point",
            "coordinates": [
                float(wloc.x),
                float(wloc.y),
            ],
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--carla-root", default="/home/ilker/simulators/CARLA_0.9.15_SOURCE")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()

    add_carla_python_paths(args.carla_root)

    import carla

    client = carla.Client(args.host, args.port)
    client.set_timeout(args.timeout)
    world = client.get_world()
    world_map = world.get_map()

    print("")
    print("CARLA bağlantısı OK")
    print("Map:", world_map.name)
    print("")
    print("Kullanım:")
    print("  Kamerayı tünel girişindeki şerit merkezine baktır.")
    print("  Komut: via_tunel_giris 1")
    print("  Kamerayı tünel içine baktır.")
    print("  Komut: via_tunel_ici 2")
    print("  Kamerayı tünel çıkışına baktır.")
    print("  Komut: via_tunel_cikis 3")
    print("  Çıkış: q")
    print("")

    while True:
        cmd = input("pick-waypoint> ").strip()

        if not cmd:
            continue

        if cmd.lower() in {"q", "quit", "exit"}:
            break

        parts = cmd.split()

        if len(parts) != 2:
            print("Kullanım: via_tunel_giris 1")
            continue

        name = parts[0]

        try:
            nokta_id = int(parts[1])
        except Exception:
            print("nokta_id sayı olmalı. Örnek: via_tunel_giris 1")
            continue

        loc, pick_mode = pick_location(world, carla)

        try:
            feature = make_feature(
                carla=carla,
                world_map=world_map,
                name=name,
                nokta_id=nokta_id,
                loc=loc,
                pick_mode=pick_mode,
            )
        except Exception as exc:
            print("HATA:", exc)
            continue

        props = feature["properties"]

        print("")
        print("KOPYALANACAK FEATURE:")
        print(json.dumps(feature, indent=2, ensure_ascii=False))
        print("")
        print(
            f"ÖZET: {props['name']} id={props['nokta_id']} "
            f"x={props['carla_x']:.2f} y={props['carla_y']:.2f} "
            f"yaw={props['carla_yaw']:.1f} road={props['road_id']} lane={props['lane_id']} "
            f"pick={props['pick_mode']}"
        )
        print("")


if __name__ == "__main__":
    main()
