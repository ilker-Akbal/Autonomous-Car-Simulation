import json
import math
import os
import sys
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from teknofest_sim.carla_loader import load_carla


class GlobalRoutePlannerNode(Node):
    """
    Yarışma uyumlu global planner.

    Girdi:
      - /adas/teknofest/mission
        Mission node sadece sıradaki görev hedefini verir.

    Çıktı:
      - /adas/planning/global_route
        CARLA GlobalRoutePlanner ile runtime'da üretilen rota.

      - /adas/planning/local_target
        RouteAgent'ın takip edeceği yakın hedef.

      - /adas/planning/route_debug
        Planner durum/debug bilgisi.

    Kritik prensip:
      GeoJSON rota değildir. GeoJSON sadece görev noktalarını verir.
      Bu node rotayı CARLA haritasından üretir.
    """

    def __init__(self):
        super().__init__("global_route_planner_node")

        self.declare_parameter("carla_root", "/home/ilker/simulators/CARLA_0.9.15_SOURCE")
        self.declare_parameter("host", "127.0.0.1")
        self.declare_parameter("port", 2000)
        self.declare_parameter("timeout", 120.0)
        self.declare_parameter("ego_role_name", "ego_vehicle")

        self.declare_parameter("mission_topic", "/adas/teknofest/mission")
        self.declare_parameter("global_route_topic", "/adas/planning/global_route")
        self.declare_parameter("local_target_topic", "/adas/planning/local_target")
        self.declare_parameter("debug_topic", "/adas/planning/route_debug")

        self.declare_parameter("route_resolution_m", 2.0)
        self.declare_parameter("lookahead_m", 20.0)
        self.declare_parameter("replan_min_interval_s", 1.0)
        self.declare_parameter("mission_timeout_s", 3.0)
        self.declare_parameter("global_route_publish_period_s", 1.0)
        self.declare_parameter("local_target_rate_hz", 10.0)

        self.declare_parameter("default_speed_hint_mps", 5.5)
        self.declare_parameter("approach_speed_hint_mps", 1.2)
        self.declare_parameter("parking_speed_hint_mps", 0.75)

        # İnsan/launch tarafı km/h. Verilirse m/s hint değerlerini override eder.
        self.declare_parameter("default_speed_hint_kmh", -1.0)
        self.declare_parameter("approach_speed_hint_kmh", -1.0)
        self.declare_parameter("parking_speed_hint_kmh", -1.0)

        self.carla_root = str(self.get_parameter("carla_root").value)
        self.host = str(self.get_parameter("host").value)
        self.port = int(self.get_parameter("port").value)
        self.timeout = float(self.get_parameter("timeout").value)
        self.ego_role_name = str(self.get_parameter("ego_role_name").value)

        self.mission_topic = str(self.get_parameter("mission_topic").value)
        self.global_route_topic = str(self.get_parameter("global_route_topic").value)
        self.local_target_topic = str(self.get_parameter("local_target_topic").value)
        self.debug_topic = str(self.get_parameter("debug_topic").value)

        self.route_resolution_m = float(self.get_parameter("route_resolution_m").value)
        self.lookahead_m = float(self.get_parameter("lookahead_m").value)
        self.replan_min_interval_s = float(self.get_parameter("replan_min_interval_s").value)
        self.mission_timeout_s = float(self.get_parameter("mission_timeout_s").value)
        self.global_route_publish_period_s = float(
            self.get_parameter("global_route_publish_period_s").value
        )
        self.local_target_rate_hz = float(self.get_parameter("local_target_rate_hz").value)

        self.default_speed_hint_mps = float(self.get_parameter("default_speed_hint_mps").value)
        self.approach_speed_hint_mps = float(self.get_parameter("approach_speed_hint_mps").value)
        self.parking_speed_hint_mps = float(self.get_parameter("parking_speed_hint_mps").value)

        self.default_speed_hint_kmh = float(self.get_parameter("default_speed_hint_kmh").value)
        self.approach_speed_hint_kmh = float(self.get_parameter("approach_speed_hint_kmh").value)
        self.parking_speed_hint_kmh = float(self.get_parameter("parking_speed_hint_kmh").value)

        if self.default_speed_hint_kmh >= 0.0:
            self.default_speed_hint_mps = self.default_speed_hint_kmh / 3.6
        if self.approach_speed_hint_kmh >= 0.0:
            self.approach_speed_hint_mps = self.approach_speed_hint_kmh / 3.6
        if self.parking_speed_hint_kmh >= 0.0:
            self.parking_speed_hint_mps = self.parking_speed_hint_kmh / 3.6

        self.carla = load_carla(self.carla_root)
        self.add_python_api_paths()

        self.client = self.carla.Client(self.host, self.port)
        self.client.set_timeout(self.timeout)
        self.world = self.client.get_world()
        self.map = self.world.get_map()
        self.planner = self.make_global_route_planner()

        self.ego = None
        self.last_ego_lookup_s = 0.0

        self.latest_mission = None
        self.last_mission_time = 0.0

        self.active_target_key = None
        self.active_target = None
        self.route_id = 0
        self.raw_route = []
        self.route_samples = []
        self.route_status = "not_planned"
        self.last_replan_time = 0.0
        self.last_global_publish_time = 0.0
        self.last_local_payload = None

        self.global_route_pub = self.create_publisher(String, self.global_route_topic, 10)
        self.local_target_pub = self.create_publisher(String, self.local_target_topic, 10)
        self.debug_pub = self.create_publisher(String, self.debug_topic, 10)

        self.create_subscription(String, self.mission_topic, self.mission_cb, 10)

        period = 1.0 / max(1.0, self.local_target_rate_hz)
        self.timer = self.create_timer(period, self.tick)

        self.get_logger().info(
            f"GlobalRoutePlannerNode hazır: map={self.map.name} "
            f"mission={self.mission_topic} global={self.global_route_topic} "
            f"local={self.local_target_topic}"
        )

    def add_python_api_paths(self):
        paths = [
            os.path.join(self.carla_root, "PythonAPI", "carla"),
            os.path.join(self.carla_root, "PythonAPI"),
            os.path.expanduser("~/CARLA_DISK/PythonAPI/carla"),
            os.path.expanduser("~/İndirilenler/PythonAPI/carla"),
        ]

        for path in paths:
            if os.path.isdir(path) and path not in sys.path:
                sys.path.append(path)

    def make_global_route_planner(self):
        try:
            from agents.navigation.global_route_planner import GlobalRoutePlanner

            try:
                return GlobalRoutePlanner(self.map, self.route_resolution_m)
            except TypeError:
                from agents.navigation.global_route_planner_dao import GlobalRoutePlannerDAO

                dao = GlobalRoutePlannerDAO(self.map, self.route_resolution_m)
                planner = GlobalRoutePlanner(dao)
                planner.setup()
                return planner

        except Exception as exc:
            raise RuntimeError(f"GlobalRoutePlanner import/kurulum hatası: {exc}") from exc

    def mission_cb(self, msg):
        try:
            self.latest_mission = json.loads(msg.data)
            self.last_mission_time = time.time()
        except Exception as exc:
            self.get_logger().warning(f"Mission parse hatası: {exc}")

    def find_ego(self):
        now = time.time()

        if self.ego is not None:
            try:
                if self.ego.is_alive:
                    return self.ego
            except Exception:
                self.ego = None

        if now - self.last_ego_lookup_s < 1.0:
            return self.ego

        self.last_ego_lookup_s = now

        try:
            vehicles = self.world.get_actors().filter("vehicle.*")
            for vehicle in vehicles:
                if vehicle.attributes.get("role_name", "") == self.ego_role_name:
                    self.ego = vehicle
                    self.get_logger().info(f"Planner ego bulundu: id={vehicle.id}")
                    return self.ego
        except Exception as exc:
            self.get_logger().warning(f"Ego arama hatası: {exc}")

        return None

    def current_ego_location(self):
        ego = self.find_ego()

        if ego is None:
            return None

        try:
            return ego.get_location()
        except Exception:
            return None

    def current_ego_transform(self):
        ego = self.find_ego()

        if ego is None:
            return None

        try:
            return ego.get_transform()
        except Exception:
            return None

    def get_objective_target(self):
        if not self.latest_mission:
            return None

        target = self.latest_mission.get("objective_target")
        if isinstance(target, dict):
            return target

        target = self.latest_mission.get("target")
        if isinstance(target, dict):
            return target

        return None

    def make_target_key(self, target):
        if not self.latest_mission or not target:
            return None

        name = str(target.get("name", ""))
        objective_index = self.latest_mission.get("objective_index", self.latest_mission.get("route_index"))
        kind = self.latest_mission.get("objective_kind", self.latest_mission.get("route_kind"))
        x = target.get("carla_x")
        y = target.get("carla_y")

        try:
            x = round(float(x), 3)
            y = round(float(y), 3)
        except Exception:
            pass

        return f"{objective_index}|{kind}|{name}|{x}|{y}"

    def target_to_location(self, target):
        x = target.get("carla_x")
        y = target.get("carla_y")
        z = target.get("carla_z", 0.2)

        if x is None or y is None:
            return None

        return self.carla.Location(x=float(x), y=float(y), z=float(z) + 0.2)

    def road_option_name(self, road_option):
        if road_option is None:
            return "UNKNOWN"

        name = getattr(road_option, "name", None)
        if name:
            return str(name).upper()

        return str(road_option).split(".")[-1].upper()

    def trace_route(self, start_loc, target_loc):
        route = self.planner.trace_route(start_loc, target_loc)

        if not route:
            raise RuntimeError("GlobalRoutePlanner trace_route boş döndü.")

        return route

    def route_to_samples(self, raw_route):
        samples = []
        total = 0.0
        prev = None

        for idx, item in enumerate(raw_route):
            wp, road_option = item
            loc = wp.transform.location

            if prev is not None:
                total += math.hypot(loc.x - prev.x, loc.y - prev.y)

            samples.append({
                "idx": idx,
                "x": float(loc.x),
                "y": float(loc.y),
                "z": float(loc.z),
                "yaw": float(wp.transform.rotation.yaw),
                "road_id": int(wp.road_id),
                "lane_id": int(wp.lane_id),
                "lane_width": float(getattr(wp, "lane_width", 0.0)),
                "is_junction": bool(getattr(wp, "is_junction", False)),
                "distance_m": float(total),
                "road_option": self.road_option_name(road_option),
            })

            prev = loc

        return samples

    def nearest_route_sample(self, loc):
        if not self.route_samples:
            return None, None

        best = None

        for s in self.route_samples:
            d = math.hypot(float(loc.x) - s["x"], float(loc.y) - s["y"])

            if best is None or d < best[0]:
                best = (d, s)

        return best

    def sample_at_distance(self, target_s):
        if not self.route_samples:
            return None

        target_s = max(0.0, min(float(target_s), float(self.route_samples[-1]["distance_m"])))

        last = self.route_samples[0]
        for sample in self.route_samples:
            if sample["distance_m"] >= target_s:
                return sample
            last = sample

        return last

    def compute_speed_hint(self, remaining_m):
        """
        Internal hız birimi m/s, loglarda km/h de yayınlanır.

        Hız kademesi artık daha erken başlar. Böylece hedefe 14 m kala
        20 km/h'den bir anda 8 km/h'ye düşmeye çalışmaz.
        """
        stage = str(self.latest_mission.get("stage", "")) if self.latest_mission else ""
        kind = str(
            self.latest_mission.get(
                "objective_kind",
                self.latest_mission.get("route_kind", ""),
            )
        ) if self.latest_mission else ""

        remaining_m = max(0.0, float(remaining_m))

        if stage == "PARKING":
            return 0.0

        # Park girişine yaklaşırken daha erken ve kademeli yavaşla.
        if kind == "park":
            if remaining_m <= 5.0:
                return min(self.default_speed_hint_mps, self.parking_speed_hint_mps)
            if remaining_m <= 12.0:
                return min(self.default_speed_hint_mps, 1.2)
            if remaining_m <= 25.0:
                return min(self.default_speed_hint_mps, 2.2)
            if remaining_m <= 45.0:
                return min(self.default_speed_hint_mps, 3.2)
            return self.default_speed_hint_mps

        # Yolcu görev noktalarına yaklaşırken de erken yavaşla.
        if kind == "task":
            if remaining_m <= 5.0:
                return min(self.default_speed_hint_mps, 0.8)
            if remaining_m <= 12.0:
                return min(self.default_speed_hint_mps, 1.4)
            if remaining_m <= 25.0:
                return min(self.default_speed_hint_mps, 2.4)
            if remaining_m <= 45.0:
                return min(self.default_speed_hint_mps, 3.4)
            return self.default_speed_hint_mps

        if remaining_m <= 8.0:
            return min(self.default_speed_hint_mps, 1.4)

        if remaining_m <= 18.0:
            return min(self.default_speed_hint_mps, 2.4)

        return self.default_speed_hint_mps

    def replan_if_needed(self):
        now = time.time()

        if self.latest_mission is None:
            self.route_status = "mission_missing"
            return False

        if now - self.last_mission_time > self.mission_timeout_s:
            self.route_status = "mission_timeout"
            return False

        target = self.get_objective_target()
        if not target:
            self.route_status = "mission_target_missing"
            return False

        target_key = self.make_target_key(target)
        if target_key is None:
            self.route_status = "target_key_missing"
            return False

        if target_key == self.active_target_key and self.route_samples:
            return True

        if now - self.last_replan_time < self.replan_min_interval_s:
            return bool(self.route_samples)

        ego_loc = self.current_ego_location()
        if ego_loc is None:
            self.route_status = "ego_missing"
            return False

        target_loc = self.target_to_location(target)
        if target_loc is None:
            self.route_status = "target_location_missing"
            return False

        try:
            raw_route = self.trace_route(ego_loc, target_loc)
            samples = self.route_to_samples(raw_route)
        except Exception as exc:
            self.route_status = f"trace_route_failed:{exc}"
            self.get_logger().error(self.route_status)
            return False

        if len(samples) < 2:
            self.route_status = "route_too_short"
            return False

        self.route_id += 1
        self.active_target_key = target_key
        self.active_target = dict(target)
        self.raw_route = raw_route
        self.route_samples = samples
        self.last_replan_time = now
        self.route_status = "route_ready"

        self.get_logger().info(
            f"GLOBAL_ROUTE_READY route_id={self.route_id} "
            f"target={target.get('name')} samples={len(samples)} "
            f"length={samples[-1]['distance_m']:.1f}m"
        )

        self.publish_global_route(force=True)
        return True

    def publish_global_route(self, force=False):
        now = time.time()

        if not self.route_samples or not self.active_target:
            return

        if (
            not force
            and now - self.last_global_publish_time < self.global_route_publish_period_s
        ):
            return

        self.last_global_publish_time = now

        payload = {
            "stamp": now,
            "route_id": self.route_id,
            "target_name": self.active_target.get("name"),
            "target": self.active_target,
            "route_status": self.route_status,
            "length_m": round(float(self.route_samples[-1]["distance_m"]), 3),
            "sample_count": len(self.route_samples),
            "points": self.route_samples,
        }

        msg = String()
        msg.data = json.dumps(payload, ensure_ascii=False)
        self.global_route_pub.publish(msg)

    def publish_local_target(self):
        if not self.route_samples or not self.active_target:
            return False

        ego_loc = self.current_ego_location()
        ego_tf = self.current_ego_transform()

        if ego_loc is None:
            self.route_status = "ego_missing_local"
            return False

        best = self.nearest_route_sample(ego_loc)
        if best is None:
            self.route_status = "nearest_route_missing"
            return False

        lateral_error_m, nearest = best
        route_len = float(self.route_samples[-1]["distance_m"])
        ego_s = float(nearest["distance_m"])
        target_s = min(route_len, ego_s + self.lookahead_m)
        local = self.sample_at_distance(target_s)

        if local is None:
            self.route_status = "local_sample_missing"
            return False

        remaining_m = max(0.0, route_len - ego_s)
        speed_hint = self.compute_speed_hint(remaining_m)

        yaw = None
        if ego_tf is not None:
            try:
                yaw = float(ego_tf.rotation.yaw)
            except Exception:
                yaw = None

        payload = {
            "stamp": time.time(),
            "route_id": self.route_id,
            "target_name": self.active_target.get("name"),
            "objective_index": self.latest_mission.get("objective_index", self.latest_mission.get("route_index"))
            if self.latest_mission else None,
            "objective_kind": self.latest_mission.get("objective_kind", self.latest_mission.get("route_kind"))
            if self.latest_mission else None,
            "mission_stage": self.latest_mission.get("stage") if self.latest_mission else None,
            "route_status": self.route_status,

            "ego_s_m": round(ego_s, 3),
            "route_length_m": round(route_len, 3),
            "remaining_m": round(remaining_m, 3),
            "lookahead_m": round(self.lookahead_m, 3),
            "lateral_error_m": round(float(lateral_error_m), 3),

            "x": local["x"],
            "y": local["y"],
            "z": local["z"],
            "yaw": local["yaw"],
            "road_id": local["road_id"],
            "lane_id": local["lane_id"],
            "road_option": local["road_option"],
            "speed_hint_mps": round(float(speed_hint), 3),
            "speed_hint_kmh": round(float(speed_hint) * 3.6, 1),

            "final_target": self.active_target,
            "ego": {
                "x": float(ego_loc.x),
                "y": float(ego_loc.y),
                "z": float(ego_loc.z),
                "yaw": yaw,
            },
        }

        msg = String()
        msg.data = json.dumps(payload, ensure_ascii=False)
        self.local_target_pub.publish(msg)
        self.last_local_payload = payload
        return True

    def publish_debug(self):
        target = self.get_objective_target() if self.latest_mission else None

        payload = {
            "stamp": time.time(),
            "route_id": self.route_id,
            "route_status": self.route_status,
            "active_target_key": self.active_target_key,
            "target_name": target.get("name") if isinstance(target, dict) else None,
            "mission_stage": self.latest_mission.get("stage") if self.latest_mission else None,
            "objective_index": self.latest_mission.get("objective_index", self.latest_mission.get("route_index"))
            if self.latest_mission else None,
            "route_sample_count": len(self.route_samples),
            "route_length_m": round(float(self.route_samples[-1]["distance_m"]), 3)
            if self.route_samples else None,
            "last_local_target": self.last_local_payload,
        }

        msg = String()
        msg.data = json.dumps(payload, ensure_ascii=False)
        self.debug_pub.publish(msg)

    def tick(self):
        ok = self.replan_if_needed()

        if ok:
            self.publish_global_route(force=False)
            self.publish_local_target()

        self.publish_debug()


def main(args=None):
    rclpy.init(args=args)
    node = GlobalRoutePlannerNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
