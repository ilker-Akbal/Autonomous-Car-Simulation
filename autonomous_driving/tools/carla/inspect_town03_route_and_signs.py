#!/usr/bin/env python3
import argparse
import json
import math
import os
import sys
import time
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


def option_name(road_option):
    if road_option is None:
        return "UNKNOWN"

    name = getattr(road_option, "name", None)
    if name:
        return str(name).upper()

    return str(road_option).split(".")[-1].upper()


def load_mission_points(path: str):
    with open(os.path.expanduser(path), "r", encoding="utf-8") as f:
        data = json.load(f)

    points = []

    for feature in data.get("features", []):
        props = feature.get("properties", {}) or {}
        geom = feature.get("geometry", {}) or {}
        coords = geom.get("coordinates", [None, None])

        name = str(props.get("name", "")).strip()
        if not name:
            continue

        x = props.get("carla_x", coords[0])
        y = props.get("carla_y", coords[1])
        z = props.get("carla_z", 0.2)

        if x is None or y is None:
            continue

        try:
            nokta_id = int(props.get("nokta_id", 9999))
        except Exception:
            nokta_id = 9999

        points.append(
            {
                "name": name,
                "nokta_id": nokta_id,
                "x": float(x),
                "y": float(y),
                "z": float(z),
                "description": str(props.get("description", name)),
            }
        )

    start = [p for p in points if p["name"].lower() == "start"]
    if not start:
        raise RuntimeError("Mission içinde start yok.")

    others = [p for p in points if p["name"].lower() != "start"]

    non_park = sorted(
        [p for p in others if not p["name"].lower().startswith("park")],
        key=lambda p: p["nokta_id"],
    )
    park = sorted(
        [p for p in others if p["name"].lower().startswith("park")],
        key=lambda p: p["nokta_id"],
    )

    ordered = start + non_park + park

    if len(ordered) < 2:
        raise RuntimeError("Route üretmek için yeterli mission noktası yok.")

    return ordered


class Town03RouteInspector:
    def __init__(self, args, carla_module):
        self.args = args
        self.carla = carla_module

        self.client = self.carla.Client(args.host, int(args.port))
        self.client.set_timeout(float(args.timeout))
        self.world = self.client.get_world()
        self.map = self.world.get_map()
        self.bp_lib = self.world.get_blueprint_library()

        self.mesh_yaw_offset_deg = float(args.mesh_yaw_offset_deg)
        self.offset_from_edge_m = float(args.offset_from_edge_m)
        self.debug_lifetime = float(args.debug_lifetime)

        print("")
        print("CARLA bağlantısı OK")
        print("Map:", self.map.name)
        print("Mission:", args.mission)
        print("Route output:", args.route_output)
        print("Signs output:", args.signs_output)
        print("")

    def make_global_route_planner(self):
        try:
            from agents.navigation.global_route_planner import GlobalRoutePlanner

            try:
                return GlobalRoutePlanner(self.map, float(self.args.resolution))
            except TypeError:
                from agents.navigation.global_route_planner_dao import GlobalRoutePlannerDAO

                dao = GlobalRoutePlannerDAO(self.map, float(self.args.resolution))
                planner = GlobalRoutePlanner(dao)
                planner.setup()
                return planner

        except Exception as exc:
            raise RuntimeError(f"GlobalRoutePlanner kurulamadı: {exc}")

    def loc(self, p):
        return self.carla.Location(x=float(p["x"]), y=float(p["y"]), z=float(p.get("z", 0.2)))

    def build_route(self, mission_points):
        planner = self.make_global_route_planner()
        route = []

        for i in range(len(mission_points) - 1):
            a = mission_points[i]
            b = mission_points[i + 1]

            start = self.loc(a)
            end = self.loc(b)

            segment = planner.trace_route(start, end)

            if not segment:
                print(f"[UYARI] Segment üretilemedi: {a['name']} -> {b['name']}")
                continue

            if route:
                segment = segment[1:]

            route.extend(segment)

            print(
                f"[ROUTE] {a['name']} -> {b['name']} "
                f"segment_wp={len(segment)}"
            )

        if len(route) < 2:
            raise RuntimeError("Route üretilemedi.")

        samples = []
        total = 0.0
        prev = None

        for idx, (wp, road_option) in enumerate(route):
            l = wp.transform.location

            if prev is not None:
                total += math.hypot(l.x - prev.x, l.y - prev.y)

            samples.append(
                {
                    "idx": idx,
                    "wp": wp,
                    "road_option": option_name(road_option),
                    "distance_m": total,
                }
            )

            prev = l

        print(f"[ROUTE READY] wp_count={len(samples)} length_m={total:.1f}")
        return samples

    def sample_at_distance(self, route_samples, distance_m):
        target = max(0.0, min(float(distance_m), route_samples[-1]["distance_m"]))
        last = route_samples[0]

        for sample in route_samples:
            if sample["distance_m"] >= target:
                return sample

            last = sample

        return last

    def same_direction_driving_lane(self, base_wp, candidate_wp):
        if candidate_wp is None:
            return False

        try:
            if candidate_wp.lane_type != self.carla.LaneType.Driving:
                return False
        except Exception:
            return False

        try:
            if int(base_wp.lane_id) * int(candidate_wp.lane_id) <= 0:
                return False
        except Exception:
            pass

        try:
            yaw_a = float(base_wp.transform.rotation.yaw)
            yaw_b = float(candidate_wp.transform.rotation.yaw)
            diff = abs((yaw_a - yaw_b + 180.0) % 360.0 - 180.0)
            if diff > 90.0:
                return False
        except Exception:
            pass

        return True

    def outermost_same_direction_lane(self, wp, side):
        current = wp
        side = str(side).upper()

        for _ in range(8):
            try:
                candidate = current.get_right_lane() if side == "R" else current.get_left_lane()
            except Exception:
                candidate = None

            if not self.same_direction_driving_lane(current, candidate):
                break

            current = candidate

        return current

    def is_on_driving_lane(self, loc):
        try:
            wp = self.map.get_waypoint(
                loc,
                project_to_road=False,
                lane_type=self.carla.LaneType.Driving,
            )
            return wp is not None
        except Exception:
            return False

    def ground_z_at(self, loc, fallback_z):
        try:
            probe = self.carla.Location(x=loc.x, y=loc.y, z=loc.z + 15.0)
            if hasattr(self.world, "ground_projection"):
                projected = self.world.ground_projection(probe, 40.0)
                if projected is not None:
                    return float(projected.location.z)
        except Exception:
            pass

        return float(fallback_z)

    def yaw_to_face_road(self, sign_loc, road_loc):
        dx = float(road_loc.x - sign_loc.x)
        dy = float(road_loc.y - sign_loc.y)
        yaw = math.degrees(math.atan2(dy, dx))
        return yaw + self.mesh_yaw_offset_deg

    def find_valid_sign_transform(self, route_samples, distance_m, side):
        side = str(side).upper()
        side_mul = 1.0 if side == "R" else -1.0

        # Önce işaretin bağlı olduğu mesafeye yakın dene.
        # Kavşak içi / yol içi olursa biraz geri/ileri kaydır.
        longitudinal_shifts = [-16.0, -12.0, -8.0, -4.0, 0.0, 4.0, 8.0, 12.0]
        outward_extra = [0.0, 0.4, 0.8, 1.2, 1.6, 2.0]

        best = None

        for ds in longitudinal_shifts:
            sample = self.sample_at_distance(route_samples, distance_m + ds)
            base_wp = sample["wp"]

            # Kavşağın içine tabela dikmeyelim.
            try:
                if bool(base_wp.is_junction):
                    continue
            except Exception:
                pass

            outer_wp = self.outermost_same_direction_lane(base_wp, side)
            outer_loc = outer_wp.transform.location
            right = outer_wp.transform.get_right_vector()

            try:
                lane_width = float(outer_wp.lane_width)
            except Exception:
                lane_width = 3.5

            for extra in outward_extra:
                lateral_m = (lane_width * 0.5) + self.offset_from_edge_m + extra

                raw_loc = self.carla.Location(
                    x=outer_loc.x + right.x * lateral_m * side_mul,
                    y=outer_loc.y + right.y * lateral_m * side_mul,
                    z=outer_loc.z + 0.4,
                )

                ground_z = self.ground_z_at(raw_loc, outer_loc.z)
                sign_loc = self.carla.Location(
                    x=raw_loc.x,
                    y=raw_loc.y,
                    z=ground_z,
                )

                on_driving = self.is_on_driving_lane(sign_loc)

                candidate = {
                    "loc": sign_loc,
                    "yaw": self.yaw_to_face_road(sign_loc, outer_loc),
                    "sample": sample,
                    "outer_wp": outer_wp,
                    "on_driving": on_driving,
                    "distance_m": sample["distance_m"],
                    "requested_distance_m": distance_m,
                    "ds": ds,
                    "lateral_m": lateral_m,
                    "lane_width": lane_width,
                    "extra": extra,
                }

                if best is None:
                    best = candidate

                if not on_driving:
                    return candidate

        # Hiç temiz nokta bulunamadıysa en iyi fallback döndürülür.
        return best

    def blueprint_for_sign(self, sign_type):
        bp_id = f"static.prop.teknofest_sign_{sign_type}"
        matches = list(self.bp_lib.filter(bp_id))
        if matches:
            return matches[0]

        try:
            return self.bp_lib.find(bp_id)
        except Exception:
            return None

    def clear_existing_teknofest_signs(self):
        destroyed = 0

        for actor in list(self.world.get_actors()):
            try:
                if str(actor.type_id).startswith("static.prop.teknofest_sign_"):
                    actor.destroy()
                    destroyed += 1
            except Exception:
                pass

        print(f"[CLEAR] destroyed teknofest signs: {destroyed}")

    def spawn_preview_sign(self, sign_feature):
        props = sign_feature["properties"]
        sign_type = props["sign_type"]
        bp = self.blueprint_for_sign(sign_type)

        if bp is None:
            print(f"[SPAWN PREVIEW FAIL] blueprint yok: {sign_type}")
            return None

        loc = self.carla.Location(
            x=float(props["carla_x"]),
            y=float(props["carla_y"]),
            z=float(props["carla_z"]) + 0.2,
        )

        rot = self.carla.Rotation(
            pitch=0.0,
            yaw=float(props["carla_yaw"]),
            roll=0.0,
        )

        actor = self.world.try_spawn_actor(bp, self.carla.Transform(loc, rot))
        if actor is None:
            print(f"[SPAWN PREVIEW FAIL] spawn olmadı: {sign_type}")
            return None

        try:
            actor.set_simulate_physics(False)
        except Exception:
            pass

        return actor

    def draw_route(self, route_samples):
        color_route = self.carla.Color(0, 255, 0)
        color_text = self.carla.Color(0, 180, 255)
        color_turn = self.carla.Color(255, 0, 0)

        for i in range(len(route_samples) - 1):
            a = route_samples[i]["wp"].transform.location + self.carla.Location(z=0.35)
            b = route_samples[i + 1]["wp"].transform.location + self.carla.Location(z=0.35)

            try:
                self.world.debug.draw_line(
                    a,
                    b,
                    thickness=0.08,
                    color=color_route,
                    life_time=self.debug_lifetime,
                    persistent_lines=False,
                )
            except Exception:
                pass

        last_label_d = -9999.0

        for sample in route_samples:
            d = sample["distance_m"]
            opt = sample["road_option"]
            loc = sample["wp"].transform.location

            if d - last_label_d >= 25.0:
                try:
                    self.world.debug.draw_string(
                        loc + self.carla.Location(z=1.2),
                        f"{sample['idx']} | {d:.0f}m",
                        draw_shadow=True,
                        color=color_text,
                        life_time=self.debug_lifetime,
                        persistent_lines=False,
                    )
                except Exception:
                    pass

                last_label_d = d

            if opt in {"LEFT", "RIGHT"}:
                try:
                    self.world.debug.draw_string(
                        loc + self.carla.Location(z=2.0),
                        opt,
                        draw_shadow=True,
                        color=color_turn,
                        life_time=self.debug_lifetime,
                        persistent_lines=False,
                    )
                except Exception:
                    pass

    def build_auto_sign_plan(self, route_samples):
        length = route_samples[-1]["distance_m"]
        plan = []

        def add(sign_type, distance_m, side, reason):
            distance_m = max(10.0, min(float(distance_m), length - 10.0))
            plan.append(
                {
                    "sign_type": sign_type,
                    "distance_m": distance_m,
                    "side": side,
                    "reason": reason,
                }
            )

        add("hiz_siniri_30", 35.0, "R", "route_start_speed")
        add("yaya_gecidi", length * 0.22, "R", "urban_crosswalk_zone")
        add("hiz_siniri_20", length * 0.42, "R", "passenger_slow_zone")
        add("dur", length * 0.64, "R", "late_stop_test")
        add("park_yeri", length - 38.0, "R", "parking_entry")
        add("park_etmek_yasaktir", length - 70.0, "L", "opposite_side_no_parking")

        # Dönüş tabelaları: GlobalRoutePlanner road_option LEFT/RIGHT verirse otomatik ekle.
        last_turn_d = -9999.0
        yol_ver_added = False

        for sample in route_samples:
            opt = sample["road_option"]
            d = sample["distance_m"]

            if opt not in {"LEFT", "RIGHT"}:
                continue

            if d < 45.0 or d > length - 45.0:
                continue

            if d - last_turn_d < 45.0:
                continue

            before_turn = d - 25.0

            if not yol_ver_added and before_turn > 50.0:
                add("yol_ver", before_turn - 18.0, "R", f"before_{opt.lower()}_junction")
                yol_ver_added = True

            if opt == "LEFT":
                add("ileriden_sola_mecburi_yon", before_turn, "R", f"auto_turn_left_at_{d:.1f}m")
            else:
                add("ileriden_saga_mecburi_yon", before_turn, "R", f"auto_turn_right_at_{d:.1f}m")

            last_turn_d = d

        # Aynı yere çok yakın işaretleri temizle.
        plan = sorted(plan, key=lambda x: x["distance_m"])

        filtered = []
        for item in plan:
            if all(abs(item["distance_m"] - old["distance_m"]) >= 14.0 for old in filtered):
                filtered.append(item)
            else:
                print(
                    f"[SKIP CLOSE SIGN] {item['sign_type']} "
                    f"d={item['distance_m']:.1f} reason={item['reason']}"
                )

        return filtered

    def route_geojson(self, route_samples):
        features = []

        for sample in route_samples:
            wp = sample["wp"]
            loc = wp.transform.location
            rot = wp.transform.rotation

            props = {
                "name": f"route_{sample['idx']:04d}",
                "kind": "route_waypoint",
                "route_index": int(sample["idx"]),
                "distance_m": round(float(sample["distance_m"]), 3),
                "road_option": sample["road_option"],
                "town": "Town03",
                "carla_x": float(loc.x),
                "carla_y": float(loc.y),
                "carla_z": float(loc.z + 0.2),
                "carla_yaw": float(rot.yaw),
                "road_id": int(wp.road_id),
                "lane_id": int(wp.lane_id),
                "lane_width": float(wp.lane_width),
                "is_junction": bool(getattr(wp, "is_junction", False)),
            }

            features.append(
                {
                    "type": "Feature",
                    "properties": props,
                    "geometry": {
                        "type": "Point",
                        "coordinates": [float(loc.x), float(loc.y)],
                    },
                }
            )

        return {
            "type": "FeatureCollection",
            "features": features,
        }

    def signs_geojson(self, route_samples, plan):
        features = []

        for idx, item in enumerate(plan, start=1):
            candidate = self.find_valid_sign_transform(
                route_samples=route_samples,
                distance_m=item["distance_m"],
                side=item["side"],
            )

            if candidate is None:
                print(f"[SIGN FAIL] {item['sign_type']} için aday bulunamadı.")
                continue

            loc = candidate["loc"]
            yaw = candidate["yaw"]
            sample = candidate["sample"]
            outer_wp = candidate["outer_wp"]
            on_driving = bool(candidate["on_driving"])

            props = {
                "name": f"sign_{idx:03d}",
                "kind": "traffic_sign",
                "sign_type": item["sign_type"],
                "side": item["side"],
                "description": f"{item['sign_type']} trafik levhası",
                "town": "Town03",
                "route_index": int(sample["idx"]),
                "route_distance_m": round(float(sample["distance_m"]), 3),
                "requested_distance_m": round(float(item["distance_m"]), 3),
                "placement_reason": item["reason"],
                "placement_on_driving_lane": on_driving,
                "placement_distance_shift_m": round(float(candidate["ds"]), 3),
                "placement_lateral_m": round(float(candidate["lateral_m"]), 3),
                "placement_extra_m": round(float(candidate["extra"]), 3),
                "nearest_road_option": sample["road_option"],
                "nearest_road_id": int(outer_wp.road_id),
                "nearest_lane_id": int(outer_wp.lane_id),
                "nearest_lane_width": float(outer_wp.lane_width),
                "carla_x": float(loc.x),
                "carla_y": float(loc.y),
                "carla_z": float(loc.z),
                "carla_yaw": float(yaw),
            }

            feature = {
                "type": "Feature",
                "properties": props,
                "geometry": {
                    "type": "Point",
                    "coordinates": [float(loc.x), float(loc.y)],
                },
            }

            features.append(feature)

            color = self.carla.Color(255, 180, 0)
            if on_driving:
                color = self.carla.Color(255, 0, 0)

            try:
                self.world.debug.draw_string(
                    loc + self.carla.Location(z=2.0),
                    f"{props['name']} {item['sign_type']}",
                    draw_shadow=True,
                    color=color,
                    life_time=self.debug_lifetime,
                    persistent_lines=False,
                )
                self.world.debug.draw_point(
                    loc + self.carla.Location(z=0.25),
                    size=0.22,
                    color=color,
                    life_time=self.debug_lifetime,
                    persistent_lines=False,
                )
            except Exception:
                pass

            status = "BAD_ON_DRIVING" if on_driving else "OK"
            print(
                f"[SIGN {status}] {props['name']} {item['sign_type']} "
                f"d={props['route_distance_m']:.1f} side={item['side']} "
                f"x={loc.x:.2f} y={loc.y:.2f} yaw={yaw:.1f} "
                f"road={props['nearest_road_id']} lane={props['nearest_lane_id']}"
            )

        return {
            "type": "FeatureCollection",
            "features": features,
        }

    def save_json(self, path, data):
        path = Path(path).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        print("[SAVE]", path)

    def run(self):
        mission_points = load_mission_points(self.args.mission)

        print("[MISSION ORDER]")
        for p in mission_points:
            print(f"  {p['nokta_id']:02d} {p['name']:12s} x={p['x']:.2f} y={p['y']:.2f}")

        route_samples = self.build_route(mission_points)

        if self.args.draw_route:
            self.draw_route(route_samples)

        plan = self.build_auto_sign_plan(route_samples)

        print("")
        print("[AUTO SIGN PLAN]")
        for item in plan:
            print(
                f"  {item['sign_type']:30s} d={item['distance_m']:.1f} "
                f"side={item['side']} reason={item['reason']}"
            )
        print("")

        route_data = self.route_geojson(route_samples)
        signs_data = self.signs_geojson(route_samples, plan)

        self.save_json(self.args.route_output, route_data)
        self.save_json(self.args.signs_output, signs_data)

        if self.args.spawn_preview:
            if self.args.clear_existing:
                self.clear_existing_teknofest_signs()

            spawned = 0
            for feature in signs_data["features"]:
                actor = self.spawn_preview_sign(feature)
                if actor is not None:
                    spawned += 1

            print(f"[PREVIEW SPAWN] spawned={spawned}/{len(signs_data['features'])}")

        print("")
        print("Bitti.")
        print("Yeşil çizgi route, turuncu noktalar tabela adaylarıdır.")
        print("Kırmızı tabela adayı görürsen o aday hâlâ driving lane üstündedir.")
        print("")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--carla-root", default="/home/ilker/simulators/CARLA_0.9.15_SOURCE")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--timeout", type=float, default=30.0)

    parser.add_argument(
        "--mission",
        default="autonomous_driving/missions/teknofest_town03_dev.geojson",
    )
    parser.add_argument(
        "--route-output",
        default="autonomous_driving/missions/teknofest_town03_inspected_route.geojson",
    )
    parser.add_argument(
        "--signs-output",
        default="autonomous_driving/missions/teknofest_town03_inspected_signs.geojson",
    )

    parser.add_argument("--resolution", type=float, default=2.0)
    parser.add_argument("--offset-from-edge-m", type=float, default=1.15)
    parser.add_argument("--mesh-yaw-offset-deg", type=float, default=-90.0)
    parser.add_argument("--debug-lifetime", type=float, default=600.0)

    parser.add_argument("--draw-route", action="store_true")
    parser.add_argument("--spawn-preview", action="store_true")
    parser.add_argument("--clear-existing", action="store_true")

    args = parser.parse_args()

    add_carla_python_paths(args.carla_root)

    try:
        import carla
    except Exception as exc:
        raise RuntimeError(
            "carla import edilemedi. --carla-root yolunu kontrol et. "
            f"Hata: {exc}"
        )

    inspector = Town03RouteInspector(args, carla)
    inspector.run()


if __name__ == "__main__":
    main()
