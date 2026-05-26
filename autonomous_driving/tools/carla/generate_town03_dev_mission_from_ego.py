import json
import math
from pathlib import Path

import carla


HOST = "127.0.0.1"
PORT = 2000

WS = Path.home() / "Masaüstü" / "Autonomous-Driving-Perception-and-Decision-System"

INPUT_MISSION = WS / "autonomous_driving" / "missions" / "teknofest_round3.geojson"
OUTPUT_MISSION = WS / "autonomous_driving" / "missions" / "teknofest_town03_dev.geojson"

STEP_M = 12.0
POINT_DISTANCE_M = 45.0
TOTAL_DISTANCE_M = 520.0


def yaw_diff(a, b):
    d = (a - b + 180.0) % 360.0 - 180.0
    return abs(d)


def find_ego(world):
    vehicles = list(world.get_actors().filter("vehicle.*"))

    for vehicle in vehicles:
        role = vehicle.attributes.get("role_name", "")
        if role in {"hero", "ego", "ego_vehicle", "autopilot"}:
            return vehicle

    if vehicles:
        return vehicles[0]

    return None


def choose_next_waypoint(current_wp, previous_yaw):
    candidates = current_wp.next(STEP_M)

    if not candidates:
        return None

    # En az ani dönüş yapan devam waypoint'ini seç.
    candidates = sorted(
        candidates,
        key=lambda wp: yaw_diff(wp.transform.rotation.yaw, previous_yaw)
    )

    return candidates[0]


def build_route_points(world):
    carla_map = world.get_map()
    ego = find_ego(world)

    if ego is not None:
        start_loc = ego.get_location()
        print(f"Ego bulundu: id={ego.id}, loc=({start_loc.x:.2f}, {start_loc.y:.2f}, {start_loc.z:.2f})")
    else:
        spawn_points = carla_map.get_spawn_points()
        if not spawn_points:
            raise RuntimeError("Town03 spawn point bulunamadı.")
        start_loc = spawn_points[0].location
        print("Ego yok. İlk spawn point kullanılacak.")

    wp = carla_map.get_waypoint(
        start_loc,
        project_to_road=True,
        lane_type=carla.LaneType.Driving,
    )

    if wp is None:
        raise RuntimeError("Başlangıç için driving waypoint bulunamadı.")

    route = []
    total = 0.0
    previous_yaw = wp.transform.rotation.yaw

    while total <= TOTAL_DISTANCE_M:
        loc = wp.transform.location
        route.append({
            "x": float(loc.x),
            "y": float(loc.y),
            "z": float(loc.z + 0.2),
            "yaw": float(wp.transform.rotation.yaw),
            "road_id": int(wp.road_id),
            "lane_id": int(wp.lane_id),
            "s": float(wp.s),
        })

        nxt = choose_next_waypoint(wp, previous_yaw)
        if nxt is None:
            print("Yol sonu geldi.")
            break

        previous_yaw = wp.transform.rotation.yaw
        wp = nxt
        total += STEP_M

    if len(route) < 8:
        raise RuntimeError(f"Route çok kısa çıktı: {len(route)} waypoint")

    print(f"Route waypoint sayısı: {len(route)}")
    return route


def sample_route(route):
    # Başlangıçtan itibaren yaklaşık 45m aralıklarla görev noktası seç.
    sampled = []

    wanted_indices = []
    spacing_steps = max(1, int(round(POINT_DISTANCE_M / STEP_M)))

    for i in range(0, len(route), spacing_steps):
        wanted_indices.append(i)

    # Mission feature sayısı fazla ise route içinden yayarak seçebilmek için listeyi uzun tut.
    for idx in wanted_indices:
        if idx < len(route):
            sampled.append(route[idx])

    if sampled[-1] != route[-1]:
        sampled.append(route[-1])

    return sampled


def geometry_type(feature):
    geom = feature.get("geometry") or {}
    return geom.get("type")


def point_coords_from_route_point(point, old_coords=None):
    # Eski dosya 2 koordinat kullanıyorsa [x, y], 3 kullanıyorsa [x, y, z] koru.
    if isinstance(old_coords, list) and len(old_coords) == 2:
        return [point["x"], point["y"]]
    return [point["x"], point["y"], point["z"]]


def update_mission(input_path, output_path, sampled, full_route):
    data = json.loads(input_path.read_text(encoding="utf-8"))

    features = data.get("features", [])
    point_features = [
        f for f in features
        if geometry_type(f) == "Point"
    ]

    if not point_features:
        raise RuntimeError("Mission içinde Point feature bulunamadı.")

    print(f"Mission Point feature sayısı: {len(point_features)}")
    print(f"Sampled route point sayısı: {len(sampled)}")

    # Point feature'ları route üstüne dağıt.
    for i, feature in enumerate(point_features):
        if len(point_features) == 1:
            route_idx = 0
        else:
            route_idx = round(i * (len(sampled) - 1) / (len(point_features) - 1))

        rp = sampled[min(route_idx, len(sampled) - 1)]
        old_coords = feature.get("geometry", {}).get("coordinates")
        feature["geometry"]["coordinates"] = point_coords_from_route_point(rp, old_coords)

        props = feature.setdefault("properties", {})
        old_name = props.get("name") or props.get("target_name") or props.get("id") or f"point_{i}"
        props["town"] = "Town03"
        props["route_index"] = int(route_idx)
        props["carla_x"] = rp["x"]
        props["carla_y"] = rp["y"]
        props["carla_z"] = rp["z"]
        props["carla_yaw"] = rp["yaw"]
        props["road_id"] = rp["road_id"]
        props["lane_id"] = rp["lane_id"]

        print(
            f"{i:02d} {old_name}: "
            f"x={rp['x']:.2f}, y={rp['y']:.2f}, z={rp['z']:.2f}, "
            f"yaw={rp['yaw']:.1f}, road={rp['road_id']}, lane={rp['lane_id']}"
        )

    # Eğer LineString varsa tüm route çizgisini de güncelle.
    for feature in features:
        if geometry_type(feature) == "LineString":
            old_coords = feature.get("geometry", {}).get("coordinates", [])
            use_2d = bool(old_coords and isinstance(old_coords[0], list) and len(old_coords[0]) == 2)

            if use_2d:
                feature["geometry"]["coordinates"] = [
                    [p["x"], p["y"]]
                    for p in full_route
                ]
            else:
                feature["geometry"]["coordinates"] = [
                    [p["x"], p["y"], p["z"]]
                    for p in full_route
                ]

            props = feature.setdefault("properties", {})
            props["town"] = "Town03"
            props["generated_from"] = "ego_waypoint_route"

    output_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("")
    print(f"Yazıldı: {output_path}")


def main():
    client = carla.Client(HOST, PORT)
    client.set_timeout(10.0)

    world = client.get_world()
    print("Map:", world.get_map().name)

    if "Town03" not in world.get_map().name:
        print("UYARI: Aktif map Town03 görünmüyor.")

    route = build_route_points(world)
    sampled = sample_route(route)

    update_mission(INPUT_MISSION, OUTPUT_MISSION, sampled, route)


if __name__ == "__main__":
    main()
