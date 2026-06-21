#!/usr/bin/env python3

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "autonomous_driving" / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from teknofest_sim.carla_loader import load_carla  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print a CARLA world snapshot for phase2 debugging.")
    parser.add_argument("--carla-root", default=None, help="CARLA installation root used to load the Python API.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--ego-role-name", default="ego_vehicle")
    parser.add_argument("--radius-m", type=float, default=30.0)
    parser.add_argument("--max-nearby", type=int, default=15)
    return parser.parse_args()


def resolve_carla_root(explicit_root: str | None) -> str:
    candidates = [
        explicit_root,
        str(REPO_ROOT / "CARLA_0.9.15"),
        "/home/ilker/simulators/CARLA_0.9.15",
        "/mnt/carla/CARLA_0.9.15",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return candidate
    raise FileNotFoundError(
        "Unable to resolve CARLA root. Pass --carla-root or place CARLA at a known path."
    )


def location_dict(location: Any) -> dict[str, float]:
    return {
        "x": round(float(location.x), 3),
        "y": round(float(location.y), 3),
        "z": round(float(location.z), 3),
    }


def find_ego(world: Any, ego_role_name: str) -> Any:
    vehicles = list(world.get_actors().filter("vehicle.*"))
    for vehicle in vehicles:
        if vehicle.attributes.get("role_name", "") in (ego_role_name, "ego", "ego_vehicle", "hero"):
            return vehicle
    return vehicles[0] if vehicles else None


def actor_distance(first: Any, second: Any) -> float:
    first_location = first.get_location()
    second_location = second.get_location()
    dx = float(first_location.x) - float(second_location.x)
    dy = float(first_location.y) - float(second_location.y)
    dz = float(first_location.z) - float(second_location.z)
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def main() -> int:
    args = parse_args()
    try:
        carla_root = resolve_carla_root(args.carla_root)
        carla = load_carla(carla_root)
        if not hasattr(carla, "Client"):
            raise RuntimeError(
                f"Resolved Python module 'carla' from {carla_root}, but it does not expose carla.Client. "
                "Check that the CARLA 0.9.15 PythonAPI egg is being loaded instead of an unrelated package."
            )
        client = carla.Client(args.host, args.port)
        client.set_timeout(5.0)
        world = client.get_world()
    except Exception as exc:
        print(f"CARLA connection failed: {exc}", file=sys.stderr)
        return 1

    world_actors = world.get_actors()
    vehicles = list(world_actors.filter("vehicle.*"))
    sensors = list(world_actors.filter("sensor.*"))
    traffic_lights = list(world_actors.filter("traffic.traffic_light*"))
    ego = find_ego(world, args.ego_role_name)

    print(f"MAP: {world.get_map().name}")
    print(f"SPAWN POINTS: {len(world.get_map().get_spawn_points())}")
    print(f"TOTAL ACTORS: {len(world_actors)}")
    print(f"VEHICLE ACTORS: {len(vehicles)}")
    print(f"SENSOR ACTORS: {len(sensors)}")
    print(f"TRAFFIC LIGHT ACTORS: {len(traffic_lights)}")

    if ego is not None:
        ego_location = ego.get_location()
        print(f"EGO: id={ego.id} role_name={ego.attributes.get('role_name', '')} location={location_dict(ego_location)}")
    else:
        print("EGO: not found")

    print("TRAFFIC LIGHTS:")
    if traffic_lights:
        for traffic_light in traffic_lights:
            try:
                state = str(traffic_light.get_state()).split(".")[-1]
            except Exception:
                state = "Unknown"
            try:
                transform = traffic_light.get_transform()
                location = location_dict(transform.location)
            except Exception:
                location = None
            print(
                f"  - id={traffic_light.id} state={state} location={location}"
            )
    else:
        print("  - none")

    print("NEARBY ACTORS AROUND EGO:")
    if ego is None:
        print("  - ego not found")
        return 0

    nearby = []
    for actor in world_actors:
        if actor.id == ego.id:
            continue
        try:
            distance_m = actor_distance(actor, ego)
            if distance_m <= args.radius_m:
                nearby.append((distance_m, actor))
        except Exception:
            continue

    if not nearby:
        print("  - none")
        return 0

    for distance_m, actor in sorted(nearby, key=lambda item: item[0])[: args.max_nearby]:
        try:
            actor_location = location_dict(actor.get_location())
        except Exception:
            actor_location = None
        print(
            f"  - distance_m={round(distance_m, 3)} id={actor.id} type_id={actor.type_id} location={actor_location}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
