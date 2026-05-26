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

    for path in paths:
        if os.path.isdir(path) and path not in sys.path:
            sys.path.append(path)


def safe_float(v, default=0.0):
    try:
        return float(v)
    except Exception:
        return default


class Town03LayoutRecorder:
    def __init__(self, carla_module, host, port, timeout, output_path):
        self.carla = carla_module
        self.client = self.carla.Client(host, int(port))
        self.client.set_timeout(float(timeout))
        self.world = self.client.get_world()
        self.map = self.world.get_map()
        self.output_path = Path(output_path).expanduser().resolve()

        self.route_features = []
        self.sign_features = []
        self.route_nokta_id = 0
        self.sign_id = 1

        print("")
        print("CARLA bağlantısı OK")
        print("Map:", self.map.name)
        print("Output:", self.output_path)
        print("")

    def spectator_transform(self):
        return self.world.get_spectator().get_transform()

    def pick_location(self):
        """
        Önce spectator kamerasının baktığı noktayı raycast ile bulmaya çalışır.
        Böylece kamerayı noktaya koymak yerine, kamerayla hedef noktaya bakıp kayıt alabiliriz.
        Raycast çalışmazsa spectator location fallback kullanılır.
        """
        spec_tf = self.spectator_transform()
        start = spec_tf.location
        fwd = spec_tf.get_forward_vector()
        end = self.carla.Location(
            x=start.x + fwd.x * 300.0,
            y=start.y + fwd.y * 300.0,
            z=start.z + fwd.z * 300.0,
        )

        try:
            hits = self.world.cast_ray(start, end)
            if hits:
                hit = hits[0]
                loc = hit.location
                return self.carla.Location(x=loc.x, y=loc.y, z=loc.z), "raycast"
        except Exception:
            pass

        return self.carla.Location(x=start.x, y=start.y, z=start.z), "spectator_location"

    def nearest_driving_waypoint(self, loc):
        return self.map.get_waypoint(
            loc,
            project_to_road=True,
            lane_type=self.carla.LaneType.Driving,
        )

    def ground_z_at(self, loc, fallback_z):
        try:
            probe = self.carla.Location(x=loc.x, y=loc.y, z=loc.z + 20.0)
            if hasattr(self.world, "ground_projection"):
                projected = self.world.ground_projection(probe, 50.0)
                if projected is not None:
                    return float(projected.location.z)
        except Exception:
            pass

        return float(fallback_z)

    def yaw_to_face_lane(self, sign_loc, lane_loc, yaw_offset=-90.0):
        dx = float(lane_loc.x - sign_loc.x)
        dy = float(lane_loc.y - sign_loc.y)
        yaw = math.degrees(math.atan2(dy, dx))
        return yaw + yaw_offset

    def draw_label(self, loc, text, color=None, z=1.8, life_time=9999.0):
        if color is None:
            color = self.carla.Color(0, 255, 80)

        try:
            self.world.debug.draw_string(
                loc + self.carla.Location(z=z),
                text,
                draw_shadow=True,
                color=color,
                life_time=life_time,
                persistent_lines=True,
            )
            self.world.debug.draw_point(
                loc + self.carla.Location(z=0.15),
                size=0.18,
                color=color,
                life_time=life_time,
                persistent_lines=True,
            )
        except Exception:
            pass

    def make_route_feature(self, name, kind, description=""):
        raw_loc, pick_mode = self.pick_location()
        wp = self.nearest_driving_waypoint(raw_loc)

        if wp is None:
            print("HATA: Yakın driving waypoint bulunamadı. Kamerayı yol/şerit üstüne hedefle.")
            return

        tf = wp.transform
        loc = tf.location
        rot = tf.rotation

        if name == "start":
            nokta_id = 0
        else:
            self.route_nokta_id += 1
            nokta_id = self.route_nokta_id

        props = {
            "name": name,
            "kind": kind,
            "description": description or name,
            "nokta_id": nokta_id,
            "town": "Town03",
            "carla_x": float(loc.x),
            "carla_y": float(loc.y),
            "carla_z": float(loc.z + 0.2),
            "carla_yaw": float(rot.yaw),
            "road_id": int(wp.road_id),
            "lane_id": int(wp.lane_id),
            "lane_width": float(wp.lane_width),
            "pick_mode": pick_mode,
        }

        feature = {
            "type": "Feature",
            "properties": props,
            "geometry": {
                "type": "Point",
                "coordinates": [float(loc.x), float(loc.y)],
            },
        }

        self.route_features.append(feature)

        self.draw_label(
            loc,
            f"{name} ({kind})",
            color=self.carla.Color(0, 255, 80),
        )

        print(
            f"KAYIT route: name={name} kind={kind} "
            f"x={loc.x:.2f} y={loc.y:.2f} yaw={rot.yaw:.1f} "
            f"road={wp.road_id} lane={wp.lane_id} pick={pick_mode}"
        )

    def make_sign_feature(self, sign_type, side="R", name=None):
        raw_loc, pick_mode = self.pick_location()

        wp = self.nearest_driving_waypoint(raw_loc)
        if wp is None:
            print("HATA: Tabelaya yakın driving lane bulunamadı. Yol kenarına yakın hedefle.")
            return

        lane_loc = wp.transform.location
        ground_z = self.ground_z_at(raw_loc, lane_loc.z)

        sign_loc = self.carla.Location(
            x=float(raw_loc.x),
            y=float(raw_loc.y),
            z=float(ground_z),
        )

        yaw = self.yaw_to_face_lane(sign_loc, lane_loc)

        if name is None:
            name = f"sign_{self.sign_id:03d}"
            self.sign_id += 1

        props = {
            "name": name,
            "kind": "traffic_sign",
            "sign_type": sign_type,
            "side": side.upper(),
            "description": f"{sign_type} trafik levhası",
            "town": "Town03",
            "carla_x": float(sign_loc.x),
            "carla_y": float(sign_loc.y),
            "carla_z": float(sign_loc.z),
            "carla_yaw": float(yaw),
            "nearest_road_id": int(wp.road_id),
            "nearest_lane_id": int(wp.lane_id),
            "nearest_lane_width": float(wp.lane_width),
            "pick_mode": pick_mode,
        }

        feature = {
            "type": "Feature",
            "properties": props,
            "geometry": {
                "type": "Point",
                "coordinates": [float(sign_loc.x), float(sign_loc.y)],
            },
        }

        self.sign_features.append(feature)

        self.draw_label(
            sign_loc,
            f"{name}: {sign_type}",
            color=self.carla.Color(255, 180, 0),
        )

        print(
            f"KAYIT sign: name={name} type={sign_type} side={side.upper()} "
            f"x={sign_loc.x:.2f} y={sign_loc.y:.2f} z={sign_loc.z:.2f} "
            f"yaw={yaw:.1f} nearest_road={wp.road_id} nearest_lane={wp.lane_id} pick={pick_mode}"
        )

    def undo(self):
        if self.sign_features:
            removed = self.sign_features.pop()
            print("UNDO sign:", removed["properties"].get("name"))
            return

        if self.route_features:
            removed = self.route_features.pop()
            print("UNDO route:", removed["properties"].get("name"))
            return

        print("UNDO: kayıt yok.")

    def list_points(self):
        print("")
        print("ROUTE POINTS")
        for i, f in enumerate(self.route_features):
            p = f["properties"]
            print(
                f"{i:02d} {p['name']:12s} kind={p['kind']:8s} "
                f"x={p['carla_x']:.2f} y={p['carla_y']:.2f} "
                f"road={p['road_id']} lane={p['lane_id']}"
            )

        print("")
        print("SIGN POINTS")
        for i, f in enumerate(self.sign_features):
            p = f["properties"]
            print(
                f"{i:02d} {p['name']:12s} type={p['sign_type']:28s} side={p['side']} "
                f"x={p['carla_x']:.2f} y={p['carla_y']:.2f} "
                f"yaw={p['carla_yaw']:.1f}"
            )
        print("")

    def save(self):
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        layout = {
            "type": "FeatureCollection",
            "features": self.route_features + self.sign_features,
        }

        route_only = {
            "type": "FeatureCollection",
            "features": self.route_features,
        }

        signs_only = {
            "type": "FeatureCollection",
            "features": self.sign_features,
        }

        route_path = self.output_path.with_name(self.output_path.stem + "_route.geojson")
        signs_path = self.output_path.with_name(self.output_path.stem + "_signs.geojson")

        self.output_path.write_text(
            json.dumps(layout, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        route_path.write_text(
            json.dumps(route_only, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        signs_path.write_text(
            json.dumps(signs_only, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        print("")
        print("KAYDEDİLDİ:")
        print("layout:", self.output_path)
        print("route :", route_path)
        print("signs :", signs_path)
        print("")

    def print_help(self):
        print("""
KOMUTLAR

  r start
      Başlangıç noktasını kaydeder.

  v via_001
      Rota ara noktası kaydeder. Düz yolda 40-60 metrede bir,
      kavşak öncesi/sonrası mutlaka via koy.

  t gorev_1
      Yolcu/görev noktası kaydeder.

  p park_giris
      Park giriş noktasını kaydeder.

  s hiz_siniri_30 R
  s yaya_gecidi R
  s yol_ver R
  s ileriden_saga_mecburi_yon R
  s ileriden_sola_mecburi_yon R
  s park_yeri R
  s park_etmek_yasaktir L
      Tabela noktası kaydeder.

  list
      Kayıtları gösterir.

  undo
      Son kaydı siler.

  save
      GeoJSON dosyalarını yazar.

  q
      Çıkış.
""")

    def loop(self):
        self.print_help()

        while True:
            try:
                cmd = input("town03-recorder> ").strip()
            except (KeyboardInterrupt, EOFError):
                print("")
                break

            if not cmd:
                continue

            parts = cmd.split()
            op = parts[0].lower()

            if op in {"q", "quit", "exit"}:
                break

            if op in {"h", "help", "?"}:
                self.print_help()
                continue

            if op == "list":
                self.list_points()
                continue

            if op == "undo":
                self.undo()
                continue

            if op == "save":
                self.save()
                continue

            if op == "r":
                if len(parts) < 2:
                    print("Kullanım: r start")
                    continue
                self.make_route_feature(parts[1], "start", "Araç başlangıç konumu")
                continue

            if op == "v":
                if len(parts) < 2:
                    print("Kullanım: v via_001")
                    continue
                self.make_route_feature(parts[1], "via", "Rota ara noktası")
                continue

            if op == "t":
                if len(parts) < 2:
                    print("Kullanım: t gorev_1")
                    continue
                self.make_route_feature(parts[1], "task", "Yolcu/görev noktası")
                continue

            if op == "p":
                name = parts[1] if len(parts) >= 2 else "park_giris"
                self.make_route_feature(name, "park", "Araç otopark giriş bölgesi noktası")
                continue

            if op == "s":
                if len(parts) < 2:
                    print("Kullanım: s hiz_siniri_30 R")
                    continue

                sign_type = parts[1]
                side = parts[2] if len(parts) >= 3 else "R"
                self.make_sign_feature(sign_type, side)
                continue

            print("Bilinmeyen komut. help yaz.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--carla-root", default="/home/ilker/simulators/CARLA_0.9.15_SOURCE")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=2000, type=int)
    parser.add_argument("--timeout", default=30.0, type=float)
    parser.add_argument(
        "--output",
        default="autonomous_driving/missions/teknofest_town03_locked_layout.geojson",
    )
    args = parser.parse_args()

    add_carla_python_paths(args.carla_root)

    try:
        import carla
    except Exception as exc:
        raise RuntimeError(
            "carla import edilemedi. --carla-root yolunu kontrol et. "
            f"Hata: {exc}"
        )

    recorder = Town03LayoutRecorder(
        carla_module=carla,
        host=args.host,
        port=args.port,
        timeout=args.timeout,
        output_path=args.output,
    )
    recorder.loop()


if __name__ == "__main__":
    main()
