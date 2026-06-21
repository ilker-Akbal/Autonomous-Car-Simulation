#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path

import carla


SIGN_TO_BLUEPRINT = {
    "hiz_siniri_20": "static.prop.teknofest_sign_hiz_siniri_20",
    "hiz_siniri_30": "static.prop.teknofest_sign_hiz_siniri_30",
    "hiz_siniri_40": "static.prop.teknofest_sign_hiz_siniri_40",
    "hiz_siniri_50": "static.prop.teknofest_sign_hiz_siniri_50",
    "yol_ver": "static.prop.teknofest_sign_yol_ver",
    "dur": "static.prop.teknofest_sign_dur",
    "saga_donulmez": "static.prop.teknofest_sign_saga_donulmez",
    "sola_donulmez": "static.prop.teknofest_sign_sola_donulmez",
    "girisi_olmayan_yol": "static.prop.teknofest_sign_girisi_olmayan_yol",
    "ileriden_saga_mecburi_yon": "static.prop.teknofest_sign_ileriden_saga_mecburi_yon",
    "ileriden_sola_mecburi_yon": "static.prop.teknofest_sign_ileriden_sola_mecburi_yon",
    "ileri_mecburi_yon": "static.prop.teknofest_sign_ileri_mecburi_yon",
    "saga_mecburi_yon": "static.prop.teknofest_sign_saga_mecburi_yon",
    "sola_mecburi_yon": "static.prop.teknofest_sign_sola_mecburi_yon",
    "yaya_gecidi": "static.prop.teknofest_sign_yaya_gecidi",
    "park_etmek_yasaktir": "static.prop.teknofest_sign_park_etmek_yasaktir",
    "park_yeri": "static.prop.teknofest_sign_park_yeri",
}


def right_vector_from_yaw(yaw_deg):
    yaw = math.radians(yaw_deg)
    return math.cos(yaw + math.pi / 2.0), math.sin(yaw + math.pi / 2.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--plan", default="autonomous_driving/missions/town03_competition_v4_sign_plan.geojson")
    ap.add_argument("--side-offset", type=float, default=2.6)
    ap.add_argument("--z-offset", type=float, default=1.2)
    ap.add_argument("--yaw-offset", type=float, default=180.0)
    ap.add_argument("--destroy-existing", action="store_true")
    args = ap.parse_args()

    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)
    world = client.get_world()
    carla_map = world.get_map()
    bps = world.get_blueprint_library()

    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    features = plan.get("features", [])

    print("MAP:", carla_map.name)
    print("PLAN:", args.plan)
    print("FEATURES:", len(features))

    if args.destroy_existing:
        old = [
            a for a in world.get_actors()
            if "static.prop.teknofest_sign" in a.type_id
        ]
        print("Destroy existing spawned signs:", len(old))
        for a in old:
            a.destroy()

    spawned = 0
    failed = 0

    for f in features:
        props = f.get("properties", {})
        geom = f.get("geometry", {})
        coords = geom.get("coordinates", [])

        sign = props.get("sign")
        sign_id = props.get("id", sign)
        side = props.get("side", "R")

        if not sign or len(coords) < 2:
            print("[SKIP] bad feature:", f)
            failed += 1
            continue

        bp_id = SIGN_TO_BLUEPRINT.get(sign)
        if not bp_id:
            print(f"[SKIP] blueprint mapping yok: {sign_id} sign={sign}")
            failed += 1
            continue

        try:
            bp = bps.find(bp_id)
        except Exception:
            print(f"[FAIL] CARLA blueprint yok: {bp_id}")
            failed += 1
            continue

        x = float(coords[0])
        y = float(coords[1])

        wp = carla_map.get_waypoint(
            carla.Location(x=x, y=y, z=2.0),
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )

        base = wp.transform
        yaw = base.rotation.yaw

        rx, ry = right_vector_from_yaw(yaw)
        side_mult = 1.0 if side.upper() == "R" else -1.0

        loc = carla.Location(
            x=x + rx * args.side_offset * side_mult,
            y=y + ry * args.side_offset * side_mult,
            z=base.location.z + args.z_offset,
        )

        rot = carla.Rotation(
            pitch=0.0,
            yaw=yaw + args.yaw_offset,
            roll=0.0,
        )

        actor = world.try_spawn_actor(bp, carla.Transform(loc, rot))

        if actor is None:
            print(f"[FAIL] spawn olmadı: {sign_id} {sign} loc=({loc.x:.2f},{loc.y:.2f},{loc.z:.2f}) yaw={rot.yaw:.1f}")
            failed += 1
        else:
            print(f"[OK] {sign_id} {sign} -> actor={actor.id} loc=({loc.x:.2f},{loc.y:.2f},{loc.z:.2f}) yaw={rot.yaw:.1f}")
            spawned += 1

    print(f"DONE spawned={spawned} failed={failed}")


if __name__ == "__main__":
    main()
