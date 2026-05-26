import math
import time
import carla


HOST = "127.0.0.1"
PORT = 2000

DISTANCE_M = 8.0
HEIGHT_M = 3.2
REFRESH_SEC = 0.03


def find_ego(world):
    vehicles = list(world.get_actors().filter("vehicle.*"))

    preferred_roles = {
        "hero",
        "ego",
        "ego_vehicle",
        "autopilot",
    }

    for vehicle in vehicles:
        role_name = vehicle.attributes.get("role_name", "")
        if role_name in preferred_roles:
            return vehicle

    # Rol adı yoksa en yeni/ilk aracı takip et
    if vehicles:
        return vehicles[0]

    return None


def look_at_rotation(from_loc, to_loc):
    dx = to_loc.x - from_loc.x
    dy = to_loc.y - from_loc.y
    dz = to_loc.z - from_loc.z

    yaw = math.degrees(math.atan2(dy, dx))
    dist_xy = math.sqrt(dx * dx + dy * dy)
    pitch = math.degrees(math.atan2(dz, dist_xy))

    return carla.Rotation(pitch=pitch, yaw=yaw, roll=0.0)


def main():
    client = carla.Client(HOST, PORT)
    client.set_timeout(10.0)

    world = client.get_world()
    spectator = world.get_spectator()

    print("Spectator follow başladı. Çıkmak için CTRL+C.")

    last_ego_id = None

    while True:
        ego = find_ego(world)

        if ego is None:
            print("Araç bulunamadı, bekleniyor...")
            time.sleep(1.0)
            continue

        if ego.id != last_ego_id:
            print(f"Takip edilen araç id={ego.id}, type={ego.type_id}, role={ego.attributes.get('role_name', '-')}")
            last_ego_id = ego.id

        tf = ego.get_transform()
        loc = tf.location
        forward = tf.get_forward_vector()

        cam_loc = carla.Location(
            x=loc.x - forward.x * DISTANCE_M,
            y=loc.y - forward.y * DISTANCE_M,
            z=loc.z + HEIGHT_M,
        )

        target_loc = carla.Location(
            x=loc.x + forward.x * 3.0,
            y=loc.y + forward.y * 3.0,
            z=loc.z + 1.2,
        )

        cam_rot = look_at_rotation(cam_loc, target_loc)
        spectator.set_transform(carla.Transform(cam_loc, cam_rot))

        time.sleep(REFRESH_SEC)


if __name__ == "__main__":
    main()
