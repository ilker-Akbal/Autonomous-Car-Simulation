#!/usr/bin/env python3
import argparse
import glob
import json
import math
import os
import select
import sys
import termios
import time
import tty
from pathlib import Path


def try_import_carla():
    try:
        import carla
        return carla
    except ImportError:
        pass

    candidates = []
    home = os.path.expanduser("~")
    roots = [
        os.environ.get("CARLA_ROOT", ""),
        f"{home}/simulators/CARLA_0.9.15",
        f"{home}/simulators/CARLA_0.9.15_SOURCE",
        f"{home}/CARLA_0.9.15",
        "/opt/carla",
    ]

    for root in roots:
        if not root:
            continue
        candidates += glob.glob(os.path.join(root, "PythonAPI", "carla", "dist", "carla-*py3*.egg"))
        candidates += glob.glob(os.path.join(root, "PythonAPI", "carla"))

    for path in candidates:
        if path not in sys.path:
            sys.path.append(path)

    try:
        import carla
        return carla
    except ImportError as e:
        print("CARLA Python API import edilemedi.")
        print("Çözüm örneği:")
        print("export CARLA_ROOT=~/simulators/CARLA_0.9.15")
        print("veya CARLA_0.9.15 yolunu script içindeki roots listesine ekle.")
        raise e


carla = try_import_carla()


def actor_role(actor):
    try:
        return actor.attributes.get("role_name", "")
    except Exception:
        return ""


def find_ego_vehicle(world, ego_id=None):
    vehicles = list(world.get_actors().filter("vehicle.*"))

    if ego_id is not None:
        for v in vehicles:
            if v.id == ego_id:
                return v
        raise RuntimeError(f"ego-id={ego_id} bulunamadı.")

    preferred_roles = {"hero", "ego", "ego_vehicle", "player", "teknofest_ego"}
    for v in vehicles:
        if actor_role(v) in preferred_roles:
            return v

    if len(vehicles) == 1:
        return vehicles[0]

    if not vehicles:
        raise RuntimeError("Dünyada vehicle.* actor yok. Önce ego aracı spawn et.")

    print("UYARI: Ego role_name bulunamadı. İlk aracı seçiyorum.")
    print("Araçlar:")
    for v in vehicles[:20]:
        print(f"  id={v.id} type={v.type_id} role={actor_role(v)}")
    return vehicles[0]


def loc_to_dict(loc):
    return {
        "x": round(float(loc.x), 6),
        "y": round(float(loc.y), 6),
        "z": round(float(loc.z), 6),
    }


def rot_to_dict(rot):
    return {
        "roll": round(float(rot.roll), 6),
        "pitch": round(float(rot.pitch), 6),
        "yaw": round(float(rot.yaw), 6),
    }


def load_json_list(path):
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data
    except Exception:
        pass
    return []


def save_json_list(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def get_front_bumper_location(vehicle, extra_offset=0.0):
    tf = vehicle.get_transform()
    loc = tf.location
    fwd = tf.get_forward_vector()

    try:
        extent_x = float(vehicle.bounding_box.extent.x)
    except Exception:
        extent_x = 2.35

    front = carla.Location(
        x=loc.x + fwd.x * (extent_x + extra_offset),
        y=loc.y + fwd.y * (extent_x + extra_offset),
        z=loc.z + fwd.z * (extent_x + extra_offset),
    )
    return front


def get_map_waypoint(world_map, location):
    try:
        return world_map.get_waypoint(
            location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
    except Exception:
        return None


def print_entry(entry):
    fb = entry["front_bumper"]
    rot = entry["rotation"]
    print("\n=== KAYDEDİLDİ ===")
    print(f"id: {entry['id']}")
    print(f"front_bumper x={fb['x']} y={fb['y']} z={fb['z']} yaw={rot['yaw']}")
    print(f"road_id={entry.get('road_id')} lane_id={entry.get('lane_id')} is_junction={entry.get('is_junction')}")
    print("Python anchor satırı:")
    print(
        "    {"
        f"\"id\": \"{entry['id']}\", "
        f"\"x\": {fb['x']}, \"y\": {fb['y']}, \"z\": {fb['z']}, "
        f"\"yaw\": {rot['yaw']}, "
        f"\"road_id\": {entry.get('road_id')}, \"lane_id\": {entry.get('lane_id')}"
        "},"
    )
    print("==================\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument(
        "--output",
        default="autonomous_driving/config/captured_tl_waypoints.json",
        help="Kaydedilecek JSON dosyası",
    )
    parser.add_argument(
        "--names",
        nargs="*",
        default=["tl_1_stop", "tl_2_stop", "tl_3_left_stop", "tl_3_right_stop"],
        help="E'ye her basışta sırayla verilecek waypoint isimleri",
    )
    parser.add_argument("--ego-id", type=int, default=None)
    parser.add_argument(
        "--front-extra-offset",
        type=float,
        default=0.0,
        help="Front bumper noktasını ekstra ileri almak için metre",
    )
    parser.add_argument(
        "--draw-life",
        type=float,
        default=600.0,
        help="CARLA debug marker kaç saniye görünsün",
    )
    args = parser.parse_args()

    output_path = Path(args.output)

    client = carla.Client(args.host, args.port)
    client.set_timeout(args.timeout)
    world = client.get_world()
    world_map = world.get_map()

    ego = find_ego_vehicle(world, args.ego_id)
    print(f"Ego seçildi: id={ego.id} type={ego.type_id} role={actor_role(ego)}")
    print(f"Çıktı dosyası: {output_path}")
    print("")
    print("Kullanım:")
    print("  e  -> mevcut front bumper noktasını waypoint olarak kaydet")
    print("  p  -> mevcut koordinatı sadece yazdır")
    print("  q  -> çık")
    print("")
    print("Not: E tuşu terminal odaktayken çalışır.")
    print("Arabayı istediğin çizgiye getir, terminale tıkla, e'ye bas.")
    print("")

    saved = load_json_list(output_path)
    capture_count = len(saved)

    old_settings = termios.tcgetattr(sys.stdin)
    tty.setcbreak(sys.stdin.fileno())

    try:
        while True:
            tf = ego.get_transform()
            center = tf.location
            front = get_front_bumper_location(ego, args.front_extra_offset)
            wp = get_map_waypoint(world_map, front)

            if select.select([sys.stdin], [], [], 0.05)[0]:
                ch = sys.stdin.read(1).lower()

                if ch == "q":
                    print("Çıkılıyor.")
                    break

                if ch not in {"e", "p"}:
                    continue

                name = (
                    args.names[capture_count]
                    if capture_count < len(args.names)
                    else f"captured_stop_{capture_count + 1}"
                )

                entry = {
                    "id": name,
                    "stamp": time.time(),
                    "vehicle_id": ego.id,
                    "vehicle_type": ego.type_id,
                    "role_name": actor_role(ego),
                    "point_type": "front_bumper",
                    "front_extra_offset_m": args.front_extra_offset,
                    "front_bumper": loc_to_dict(front),
                    "vehicle_center": loc_to_dict(center),
                    "rotation": rot_to_dict(tf.rotation),
                    "road_id": int(wp.road_id) if wp else None,
                    "lane_id": int(wp.lane_id) if wp else None,
                    "lane_width": round(float(wp.lane_width), 6) if wp else None,
                    "is_junction": bool(wp.is_junction) if wp else None,
                }

                if ch == "p":
                    print_entry(entry)
                    continue

                saved.append(entry)
                save_json_list(output_path, saved)
                capture_count += 1

                world.debug.draw_point(
                    front,
                    size=0.22,
                    color=carla.Color(0, 255, 0),
                    life_time=args.draw_life,
                )
                world.debug.draw_string(
                    front + carla.Location(z=0.7),
                    name,
                    draw_shadow=False,
                    color=carla.Color(0, 255, 0),
                    life_time=args.draw_life,
                )

                print_entry(entry)

    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)


if __name__ == "__main__":
    main()
