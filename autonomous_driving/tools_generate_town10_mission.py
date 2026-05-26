import json
from pathlib import Path

import carla


HOST = "127.0.0.1"
PORT = 2000
SPAWN_INDEX = 0
OUT_PATH = Path("missions/teknofest_round3_town10_current.geojson")


def to_feature(name, description, nokta_id, geo, yaw):
    return {
        "type": "Feature",
        "properties": {
            "name": name,
            "description": description,
            "nokta_id": nokta_id,
            "yaw": float(yaw),
        },
        "geometry": {
            "type": "Point",
            "coordinates": [
                float(geo.longitude),
                float(geo.latitude),
            ],
        },
    }


def main():
    client = carla.Client(HOST, PORT)
    client.set_timeout(120.0)

    world = client.get_world()
    carla_map = world.get_map()

    print("Connected map:", carla_map.name)

    spawn_points = carla_map.get_spawn_points()
    if not spawn_points:
        raise RuntimeError("CARLA haritasında spawn point bulunamadı.")

    spawn_index = max(0, min(SPAWN_INDEX, len(spawn_points) - 1))
    start_tf = spawn_points[spawn_index]

    start_wp = carla_map.get_waypoint(
        start_tf.location,
        project_to_road=True,
        lane_type=carla.LaneType.Driving,
    )

    if start_wp is None:
        raise RuntimeError("Spawn point için waypoint bulunamadı.")

    mission_points = [
        ("start", "Araç başlangıç konumu", 0, 0.0),
        ("gorev_1", "1. yolcu alma/bindirme noktası", 1, 35.0),
        ("gorev_2", "2. yolcu indirme/bindirme noktası", 2, 70.0),
        ("park_giris", "Araç otopark giriş bölgesi noktası", 3, 105.0),
    ]

    features = []

    for name, desc, nokta_id, dist in mission_points:
        if dist == 0.0:
            wp = start_wp
        else:
            candidates = start_wp.next(dist)
            if not candidates:
                raise RuntimeError(f"{name} için {dist} metre ileride waypoint bulunamadı.")
            wp = candidates[0]

        tf = wp.transform
        geo = carla_map.transform_to_geolocation(tf.location)

        features.append(to_feature(name, desc, nokta_id, geo, tf.rotation.yaw))

        print(
            f"{name:10s} | dist={dist:6.1f}m | "
            f"lat={geo.latitude:.8f} lon={geo.longitude:.8f} yaw={tf.rotation.yaw:.2f}"
        )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "type": "FeatureCollection",
        "features": features,
    }

    OUT_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nWrote:", OUT_PATH.resolve())


if __name__ == "__main__":
    main()
