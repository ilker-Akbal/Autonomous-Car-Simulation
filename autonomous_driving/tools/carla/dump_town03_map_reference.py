#!/usr/bin/env python3
import argparse
import csv
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

    for path in paths:
        if os.path.isdir(path) and path not in sys.path:
            sys.path.append(path)


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


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

    return start + non_park + park


class Town03MapDumper:
    def __init__(self, args, carla):
        self.args = args
        self.carla = carla

        self.client = self.carla.Client(args.host, int(args.port))
        self.client.set_timeout(float(args.timeout))

        self.world = self.client.get_world()
        self.map = self.world.get_map()
        self.out_dir = Path(args.out_dir).expanduser().resolve()
        ensure_dir(self.out_dir)

        self.summary_lines = []

        self.log("")
        self.log("CARLA bağlantısı OK")
        self.log(f"Map: {self.map.name}")
        self.log(f"Output dir: {self.out_dir}")
        self.log("")

    def log(self, text=""):
        print(text)
        self.summary_lines.append(str(text))

    def save_json(self, filename: str, data: dict):
        path = self.out_dir / filename
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        self.log(f"[SAVE] {path}")
        return path

    def save_csv(self, filename: str, rows: list, fieldnames: list):
        path = self.out_dir / filename

        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=fieldnames,
                extrasaction="ignore",
            )
            writer.writeheader()

            for row in rows:
                safe_row = {key: row.get(key, "") for key in fieldnames}
                writer.writerow(safe_row)

        self.log(f"[SAVE] {path}")
        return path

    def waypoint_info_at_location(self, loc):
        wp = self.map.get_waypoint(
            loc,
            project_to_road=True,
            lane_type=self.carla.LaneType.Driving,
        )

        if wp is None:
            return None

        return {
            "road_id": int(wp.road_id),
            "section_id": int(getattr(wp, "section_id", -1)),
            "lane_id": int(wp.lane_id),
            "lane_width": float(wp.lane_width),
            "s": float(getattr(wp, "s", 0.0)),
            "is_junction": bool(getattr(wp, "is_junction", False)),
            "wp_x": float(wp.transform.location.x),
            "wp_y": float(wp.transform.location.y),
            "wp_z": float(wp.transform.location.z),
            "wp_yaw": float(wp.transform.rotation.yaw),
        }

    def transform_feature(self, name, kind, transform, extra=None):
        loc = transform.location
        rot = transform.rotation

        props = {
            "name": name,
            "kind": kind,
            "town": "Town03",
            "carla_x": float(loc.x),
            "carla_y": float(loc.y),
            "carla_z": float(loc.z),
            "carla_yaw": float(rot.yaw),
            "pitch": float(rot.pitch),
            "roll": float(rot.roll),
        }

        wp_info = self.waypoint_info_at_location(loc)
        if wp_info:
            props.update(wp_info)

        if extra:
            props.update(extra)

        return {
            "type": "Feature",
            "properties": props,
            "geometry": {
                "type": "Point",
                "coordinates": [float(loc.x), float(loc.y)],
            },
        }

    def dump_spawn_points(self):
        spawn_points = list(self.map.get_spawn_points())

        features = []
        rows = []

        self.log("")
        self.log(f"[SPAWN POINTS] count={len(spawn_points)}")

        for i, sp in enumerate(spawn_points):
            name = f"SP_{i:03d}"

            feature = self.transform_feature(
                name=name,
                kind="spawn_point",
                transform=sp,
                extra={"spawn_index": i},
            )
            features.append(feature)

            props = feature["properties"]

            row = {
                "spawn_index": i,
                "name": name,
                "x": props.get("carla_x"),
                "y": props.get("carla_y"),
                "z": props.get("carla_z"),
                "yaw": props.get("carla_yaw"),
                "road_id": props.get("road_id"),
                "lane_id": props.get("lane_id"),
                "lane_width": props.get("lane_width"),
                "is_junction": props.get("is_junction"),
            }
            rows.append(row)

            if self.args.draw_spawn_points:
                color = self.carla.Color(0, 180, 255)

                try:
                    self.world.debug.draw_string(
                        sp.location + self.carla.Location(z=2.0),
                        f"{name}\nroad={row['road_id']} lane={row['lane_id']}",
                        draw_shadow=True,
                        color=color,
                        life_time=float(self.args.debug_lifetime),
                        persistent_lines=False,
                    )
                    self.world.debug.draw_point(
                        sp.location + self.carla.Location(z=0.35),
                        size=0.20,
                        color=color,
                        life_time=float(self.args.debug_lifetime),
                        persistent_lines=False,
                    )
                except Exception:
                    pass

        data = {
            "type": "FeatureCollection",
            "features": features,
        }

        self.save_json("town03_spawn_points.geojson", data)
        self.save_csv(
            "town03_spawn_points.csv",
            rows,
            [
                "spawn_index",
                "name",
                "x",
                "y",
                "z",
                "yaw",
                "road_id",
                "lane_id",
                "lane_width",
                "is_junction",
            ],
        )

        self.log("İlk 20 spawn point:")
        for row in rows[:20]:
            self.log(
                f"  {row['name']} x={row['x']:.2f} y={row['y']:.2f} "
                f"yaw={row['yaw']:.1f} road={row['road_id']} lane={row['lane_id']}"
            )

        return features

    def dump_waypoints(self):
        step = float(self.args.waypoint_step)
        waypoints = list(self.map.generate_waypoints(step))

        features = []
        rows = []

        self.log("")
        self.log(f"[WAYPOINTS] step={step}m count={len(waypoints)}")

        for i, wp in enumerate(waypoints):
            loc = wp.transform.location
            rot = wp.transform.rotation

            name = f"WP_{i:05d}"

            props = {
                "name": name,
                "kind": "driving_waypoint",
                "town": "Town03",
                "wp_index": i,
                "carla_x": float(loc.x),
                "carla_y": float(loc.y),
                "carla_z": float(loc.z),
                "carla_yaw": float(rot.yaw),
                "road_id": int(wp.road_id),
                "section_id": int(getattr(wp, "section_id", -1)),
                "lane_id": int(wp.lane_id),
                "lane_width": float(wp.lane_width),
                "s": float(getattr(wp, "s", 0.0)),
                "is_junction": bool(getattr(wp, "is_junction", False)),
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

            rows.append(
                {
                    "wp_index": i,
                    "name": name,
                    "x": props["carla_x"],
                    "y": props["carla_y"],
                    "z": props["carla_z"],
                    "yaw": props["carla_yaw"],
                    "road_id": props["road_id"],
                    "section_id": props["section_id"],
                    "lane_id": props["lane_id"],
                    "lane_width": props["lane_width"],
                    "s": props["s"],
                    "is_junction": props["is_junction"],
                }
            )

            if self.args.draw_waypoints and i % int(self.args.draw_waypoint_every) == 0:
                color = self.carla.Color(255, 255, 0)

                try:
                    self.world.debug.draw_string(
                        loc + self.carla.Location(z=1.2),
                        f"{name}\nroad={wp.road_id} lane={wp.lane_id}",
                        draw_shadow=True,
                        color=color,
                        life_time=float(self.args.debug_lifetime),
                        persistent_lines=False,
                    )
                except Exception:
                    pass

        data = {
            "type": "FeatureCollection",
            "features": features,
        }

        self.save_json(f"town03_waypoints_{int(step)}m.geojson", data)
        self.save_csv(
            f"town03_waypoints_{int(step)}m.csv",
            rows,
            [
                "wp_index",
                "name",
                "x",
                "y",
                "z",
                "yaw",
                "road_id",
                "section_id",
                "lane_id",
                "lane_width",
                "s",
                "is_junction",
            ],
        )

        return features

    def dump_topology(self):
        topology = list(self.map.get_topology())

        features = []

        self.log("")
        self.log(f"[TOPOLOGY] segments={len(topology)}")

        for i, pair in enumerate(topology):
            wp_a, wp_b = pair
            loc_a = wp_a.transform.location
            loc_b = wp_b.transform.location

            props = {
                "name": f"TOPO_{i:04d}",
                "kind": "topology_segment",
                "town": "Town03",
                "segment_index": i,
                "start_road_id": int(wp_a.road_id),
                "start_lane_id": int(wp_a.lane_id),
                "start_s": float(getattr(wp_a, "s", 0.0)),
                "end_road_id": int(wp_b.road_id),
                "end_lane_id": int(wp_b.lane_id),
                "end_s": float(getattr(wp_b, "s", 0.0)),
            }

            feature = {
                "type": "Feature",
                "properties": props,
                "geometry": {
                    "type": "LineString",
                    "coordinates": [
                        [float(loc_a.x), float(loc_a.y)],
                        [float(loc_b.x), float(loc_b.y)],
                    ],
                },
            }

            features.append(feature)

        data = {
            "type": "FeatureCollection",
            "features": features,
        }

        self.save_json("town03_topology.geojson", data)
        return features

    def make_global_route_planner(self):
        try:
            from agents.navigation.global_route_planner import GlobalRoutePlanner

            try:
                return GlobalRoutePlanner(self.map, float(self.args.route_resolution))
            except TypeError:
                from agents.navigation.global_route_planner_dao import GlobalRoutePlannerDAO

                dao = GlobalRoutePlannerDAO(self.map, float(self.args.route_resolution))
                planner = GlobalRoutePlanner(dao)
                planner.setup()
                return planner

        except Exception as exc:
            raise RuntimeError(f"GlobalRoutePlanner kurulamadı: {exc}")

    def dump_current_mission_route(self):
        mission_path = str(self.args.mission)

        if not mission_path or not os.path.exists(os.path.expanduser(mission_path)):
            self.log("")
            self.log("[MISSION ROUTE] mission dosyası yok, route çıkarılmadı.")
            return []

        mission_points = load_mission_points(mission_path)

        self.log("")
        self.log("[MISSION ORDER]")
        for p in mission_points:
            self.log(
                f"  {p['nokta_id']:02d} {p['name']:12s} "
                f"x={p['x']:.2f} y={p['y']:.2f} z={p['z']:.2f}"
            )

        planner = self.make_global_route_planner()
        raw_route = []

        for i in range(len(mission_points) - 1):
            a = mission_points[i]
            b = mission_points[i + 1]

            start = self.carla.Location(x=a["x"], y=a["y"], z=a["z"])
            end = self.carla.Location(x=b["x"], y=b["y"], z=b["z"])

            segment = planner.trace_route(start, end)

            self.log(
                f"[ROUTE SEGMENT] {a['name']} -> {b['name']} "
                f"wp_count={len(segment)}"
            )

            if raw_route and segment:
                segment = segment[1:]

            raw_route.extend(segment)

        if len(raw_route) < 2:
            self.log("[MISSION ROUTE] route üretilemedi.")
            return []

        features = []
        rows = []

        total = 0.0
        prev = None

        for i, item in enumerate(raw_route):
            wp, road_option = item
            loc = wp.transform.location
            rot = wp.transform.rotation

            if prev is not None:
                total += math.hypot(loc.x - prev.x, loc.y - prev.y)

            opt = option_name(road_option)

            props = {
                "name": f"ROUTE_{i:04d}",
                "kind": "mission_route_waypoint",
                "town": "Town03",
                "route_index": i,
                "distance_m": round(float(total), 3),
                "road_option": opt,
                "carla_x": float(loc.x),
                "carla_y": float(loc.y),
                "carla_z": float(loc.z),
                "carla_yaw": float(rot.yaw),
                "road_id": int(wp.road_id),
                "section_id": int(getattr(wp, "section_id", -1)),
                "lane_id": int(wp.lane_id),
                "lane_width": float(wp.lane_width),
                "s": float(getattr(wp, "s", 0.0)),
                "is_junction": bool(getattr(wp, "is_junction", False)),
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
            rows.append(props)

            if self.args.draw_route:
                color = self.carla.Color(0, 255, 0)

                if prev is not None:
                    try:
                        self.world.debug.draw_line(
                            self.carla.Location(x=prev.x, y=prev.y, z=prev.z + 0.35),
                            loc + self.carla.Location(z=0.35),
                            thickness=0.08,
                            color=color,
                            life_time=float(self.args.debug_lifetime),
                            persistent_lines=False,
                        )
                    except Exception:
                        pass

                if i % int(self.args.draw_route_label_every) == 0:
                    try:
                        self.world.debug.draw_string(
                            loc + self.carla.Location(z=1.5),
                            f"R_{i:04d}\n{total:.0f}m\n{opt}",
                            draw_shadow=True,
                            color=self.carla.Color(0, 255, 0),
                            life_time=float(self.args.debug_lifetime),
                            persistent_lines=False,
                        )
                    except Exception:
                        pass

            prev = loc

        data = {
            "type": "FeatureCollection",
            "features": features,
        }

        self.log(f"[MISSION ROUTE READY] wp_count={len(features)} length_m={total:.1f}")

        self.save_json("town03_current_mission_route.geojson", data)
        self.save_csv(
            "town03_current_mission_route.csv",
            rows,
            [
                "name",
                "kind",
                "route_index",
                "distance_m",
                "road_option",
                "carla_x",
                "carla_y",
                "carla_z",
                "carla_yaw",
                "road_id",
                "section_id",
                "lane_id",
                "lane_width",
                "s",
                "is_junction",
            ],
        )

        return features

    def dump_opendrive(self):
        if not self.args.export_opendrive:
            return

        path = self.out_dir / "town03_map.opendrive.xodr"

        try:
            xodr = self.map.to_opendrive()
            path.write_text(xodr, encoding="utf-8")
            self.log(f"[SAVE] {path}")
        except Exception as exc:
            self.log(f"[WARN] OpenDRIVE export başarısız: {exc}")

    def save_summary(self):
        path = self.out_dir / "town03_reference_summary.txt"
        path.write_text("\n".join(self.summary_lines), encoding="utf-8")
        print(f"[SAVE] {path}")

    def run(self):
        self.dump_spawn_points()
        self.dump_waypoints()
        self.dump_topology()
        self.dump_current_mission_route()
        self.dump_opendrive()
        self.save_summary()

        self.log("")
        self.log("BİTTİ.")
        self.log("Bana özellikle şu dosyaları at:")
        self.log("  town03_spawn_points.csv")
        self.log("  town03_waypoints_10m.csv")
        self.log("  town03_current_mission_route.csv")
        self.log("  town03_reference_summary.txt")
        self.log("")
        self.log("Tünel tarafında ekranda gördüğün SP_XXX numaralarını da yazarsan")
        self.log("ben direkt mission dosyasına via_tunel_* noktalarını eklerim.")


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
        "--out-dir",
        default="autonomous_driving/outputs/town03_map_reference",
    )

    parser.add_argument("--waypoint-step", type=float, default=10.0)
    parser.add_argument("--route-resolution", type=float, default=2.0)

    parser.add_argument("--draw-spawn-points", action="store_true")
    parser.add_argument("--draw-waypoints", action="store_true")
    parser.add_argument("--draw-route", action="store_true")
    parser.add_argument("--draw-waypoint-every", type=int, default=25)
    parser.add_argument("--draw-route-label-every", type=int, default=15)
    parser.add_argument("--debug-lifetime", type=float, default=1800.0)

    parser.add_argument("--export-opendrive", action="store_true")

    args = parser.parse_args()

    add_carla_python_paths(args.carla_root)

    try:
        import carla
    except Exception as exc:
        raise RuntimeError(
            "carla import edilemedi. --carla-root yolunu kontrol et. "
            f"Hata: {exc}"
        )

    dumper = Town03MapDumper(args, carla)
    dumper.run()


if __name__ == "__main__":
    main()
