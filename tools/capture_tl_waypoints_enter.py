#!/usr/bin/env python3
import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path


def import_carla():
    try:
        import carla
        return carla
    except ImportError:
        root = os.path.expanduser(os.environ.get("CARLA_ROOT", "~/simulators/CARLA_0.9.15"))
        eggs = glob.glob(root + "/PythonAPI/carla/dist/carla-*py3*.egg")
        if eggs:
            sys.path.append(eggs[0])
        import carla
        return carla


carla = import_carla()


def role(actor):
    return actor.attributes.get("role_name", "")


def find_vehicle(world, ego_id=None):
    vehicles = list(world.get_actors().filter("vehicle.*"))

    if ego_id is not None:
        for v in vehicles:
            if v.id == ego_id:
                return v
        raise RuntimeError(f"ego-id={ego_id} bulunamadı")

    for v in vehicles:
        if role(v) == "hero":
            return v

    if len(vehicles) == 1:
        return vehicles[0]

    print("Birden fazla araç var. Şunlardan ego-id seç:")
    for v in vehicles:
        tf = v.get_transform()
        print(f"id={v.id} type={v.type_id} role={role(v)} x={tf.location.x:.2f} y={tf.location.y:.2f}")
    raise RuntimeError("--ego-id vermen lazım")


def front_bumper(vehicle):
    tf = vehicle.get_transform()
    loc = tf.location
    fwd = tf.get_forward_vector()
    extent = float(vehicle.bounding_box.extent.x)

    return carla.Location(
        x=loc.x + fwd.x * extent,
        y=loc.y + fwd.y * extent,
        z=loc.z + fwd.z * extent,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--ego-id", type=int, default=None)
    ap.add_argument("--output", default="autonomous_driving/config/captured_tl_waypoints.json")
    ap.add_argument("--names", nargs="*", default=["tl_1_stop", "tl_2_stop", "tl_3_left_stop", "tl_3_right_stop"])
    args = ap.parse_args()

    client = carla.Client(args.host, args.port)
    client.set_timeout(5.0)
    world = client.get_world()
    world_map = world.get_map()

    vehicle = find_vehicle(world, args.ego_id)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    saved = []
    if out.exists():
        try:
            saved = json.loads(out.read_text(encoding="utf-8"))
            if not isinstance(saved, list):
                saved = []
        except Exception:
            saved = []

    print("")
    print(f"Ego seçildi: id={vehicle.id} type={vehicle.type_id} role={role(vehicle)}")
    print(f"Kayıt dosyası: {out}")
    print("")
    print("KULLANIM:")
    print("Arabayı istediğin yere çek.")
    print("Bu terminale tıkla.")
    print("ENTER'a basınca waypoint kaydeder.")
    print("Çıkmak için CTRL+C.")
    print("")

    while True:
        idx = len(saved)
        name = args.names[idx] if idx < len(args.names) else f"captured_stop_{idx + 1}"

        input(f"[{idx + 1}] {name} kaydetmek için ENTER'a bas... ")

        tf = vehicle.get_transform()
        fb = front_bumper(vehicle)

        try:
            wp = world_map.get_waypoint(fb, project_to_road=True, lane_type=carla.LaneType.Driving)
            road_id = int(wp.road_id)
            lane_id = int(wp.lane_id)
            is_junction = bool(wp.is_junction)
        except Exception:
            road_id = None
            lane_id = None
            is_junction = None

        entry = {
            "id": name,
            "stamp": time.time(),
            "vehicle_id": vehicle.id,
            "vehicle_type": vehicle.type_id,
            "role_name": role(vehicle),
            "point_type": "front_bumper",
            "x": round(float(fb.x), 6),
            "y": round(float(fb.y), 6),
            "z": round(float(fb.z), 6),
            "yaw": round(float(tf.rotation.yaw), 6),
            "road_id": road_id,
            "lane_id": lane_id,
            "is_junction": is_junction,
        }

        saved.append(entry)
        out.write_text(json.dumps(saved, indent=2, ensure_ascii=False), encoding="utf-8")

        world.debug.draw_point(fb, size=0.25, color=carla.Color(0, 255, 0), life_time=600.0)
        world.debug.draw_string(
            fb + carla.Location(z=0.8),
            name,
            draw_shadow=False,
            color=carla.Color(0, 255, 0),
            life_time=600.0,
        )

        print("")
        print("=== KAYDEDİLDİ ===")
        print(f'id: {name}')
        print(f'x={entry["x"]} y={entry["y"]} z={entry["z"]} yaw={entry["yaw"]}')
        print(f'road_id={road_id} lane_id={lane_id} is_junction={is_junction}')
        print("Python anchor satırı:")
        print(f'{{"id": "{name}", "x": {entry["x"]}, "y": {entry["y"]}, "z": {entry["z"]}, "yaw": {entry["yaw"]}, "road_id": {road_id}, "lane_id": {lane_id}}},')
        print("==================")
        print("")


if __name__ == "__main__":
    main()
