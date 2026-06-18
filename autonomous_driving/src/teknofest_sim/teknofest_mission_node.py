import json
import math
import time
from dataclasses import asdict

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import NavSatFix
from std_msgs.msg import String

from teknofest_common.runtime_logging import RuntimeJsonlLogger
from teknofest_sim.carla_loader import load_carla
from teknofest_sim.geojson_mission import (
    haversine_meters,
    load_mission_geojson,
    mission_to_dict,
)


class TeknofestMissionNode(Node):
    """
    TEKNOFEST mission state node.

    Yarışma modu:
      - GeoJSON rota dosyası değildir.
      - GeoJSON sadece start, gorev_* / passenger_* ve park_giris içerir.
      - via_* noktaları yarışma modunda kabul edilmez.
      - Rota üretme işi mission node'un görevi değildir.
      - Mission node sadece sıradaki görev hedefini ve görev durumunu yayınlar.

    Eski/debug modu:
      - competition_mode=False yapılırsa eski via_* sıralı rota mantığı çalışabilir.
      - Ana yarışma/simülasyon akışında bu kullanılmamalıdır.
    """

    def __init__(self):
        super().__init__("teknofest_mission_node")

        self.declare_parameter("mission_geojson", "missions/teknofest_round3.geojson")
        self.declare_parameter("round_name", "round_3")

        # Final yarışma mantığında True kalmalı.
        self.declare_parameter("competition_mode", True)

        self.declare_parameter("gnss_topic", "/adas/localization/gnss")
        self.declare_parameter("mission_topic", "/adas/teknofest/mission")
        self.declare_parameter("event_topic", "/adas/teknofest/events")

        # Şartname görev noktası toleransı 1 m.
        # Simülasyonda sensör/BasicAgent dalgalanması için launch tarafında 2-3 m verilebilir.
        self.declare_parameter("point_pass_tolerance_m", 1.0)

        # TL_ROUTE_MISSION_FIX_V2:
        # Launch'ta point_pass_tolerance 3m verilince ışık/kavşak çevresinde yolcu durağı
        # erken başlıyor. Görev/task duruşu için daha sıkı tolerans kullan.
        self.declare_parameter("task_stop_tolerance_m", 1.15)
        self.declare_parameter("park_entry_tolerance_m", -1.0)
        self.declare_parameter("distance_reference", "front_bumper")
        self.declare_parameter("front_bumper_offset_m", -1.0)

        # Yolcu indirme/bindirme şartname aralığı.
        self.declare_parameter("passenger_stop_min_s", 15.0)
        self.declare_parameter("passenger_stop_max_s", 20.0)

        # Park bölgesine ulaştıktan sonra 3 dakika.
        self.declare_parameter("park_time_limit_s", 180.0)

        # Şimdilik park planner gelene kadar park girişinden sonra kısa bekleme ile tamamlıyor.
        # Parking planner eklendiğinde bu değer kullanılmayacak.
        self.declare_parameter("temporary_parking_complete_s", 8.0)

        # Town03 CARLA local coordinate distance.
        self.declare_parameter("use_carla_xy_distance", True)
        self.declare_parameter("carla_root", "/home/ilker/simulators/CARLA_0.9.15_SOURCE")
        self.declare_parameter("host", "127.0.0.1")
        self.declare_parameter("port", 2000)
        self.declare_parameter("timeout", 120.0)
        self.declare_parameter("ego_role_name", "ego_vehicle")
        self.declare_parameter("log_root", "autonomous_driving/outputs/teknofest_sim_logs")
        self.declare_parameter("log_session_id", "")
        self.declare_parameter("jsonl_logging_enabled", True)
        self.declare_parameter("ros_log_period_s", 1.0)

        self.mission_geojson = str(self.get_parameter("mission_geojson").value)
        self.round_name = str(self.get_parameter("round_name").value)
        self.competition_mode = bool(self.get_parameter("competition_mode").value)

        self.gnss_topic = str(self.get_parameter("gnss_topic").value)
        self.mission_topic = str(self.get_parameter("mission_topic").value)
        self.event_topic = str(self.get_parameter("event_topic").value)

        self.point_pass_tolerance_m = float(self.get_parameter("point_pass_tolerance_m").value)
        self.task_stop_tolerance_m = float(self.get_parameter("task_stop_tolerance_m").value)
        self.park_entry_tolerance_m = float(self.get_parameter("park_entry_tolerance_m").value)
        if self.park_entry_tolerance_m < 0.0:
            self.park_entry_tolerance_m = self.point_pass_tolerance_m
        self.distance_reference = str(self.get_parameter("distance_reference").value)
        self.front_bumper_offset_m = float(self.get_parameter("front_bumper_offset_m").value)
        self.passenger_stop_min_s = float(self.get_parameter("passenger_stop_min_s").value)
        self.passenger_stop_max_s = float(self.get_parameter("passenger_stop_max_s").value)
        self.park_time_limit_s = float(self.get_parameter("park_time_limit_s").value)
        self.temporary_parking_complete_s = float(
            self.get_parameter("temporary_parking_complete_s").value
        )

        self.use_carla_xy_distance = bool(self.get_parameter("use_carla_xy_distance").value)
        self.carla_root = str(self.get_parameter("carla_root").value)
        self.host = str(self.get_parameter("host").value)
        self.port = int(self.get_parameter("port").value)
        self.timeout = float(self.get_parameter("timeout").value)
        self.ego_role_name = str(self.get_parameter("ego_role_name").value)
        self.ros_log_period_s = float(self.get_parameter("ros_log_period_s").value)

        self.mission = load_mission_geojson(
            self.mission_geojson,
            self.round_name,
            competition_mode=self.competition_mode,
        )

        self.current_lat = None
        self.current_lon = None

        self.carla = None
        self.client = None
        self.world = None
        self.ego = None
        self.last_ego_lookup_s = 0.0
        self.last_ros_log_s = 0.0
        self.last_distance_reference_used = "unknown"

        self.runtime_logger = RuntimeJsonlLogger(
            node_name="teknofest_mission_node",
            file_name="mission_runtime.jsonl",
            log_root=str(self.get_parameter("log_root").value),
            session_id=str(self.get_parameter("log_session_id").value) or None,
            enabled=bool(self.get_parameter("jsonl_logging_enabled").value),
        )

        if self.use_carla_xy_distance:
            self.connect_to_carla()

        self.stage = "GO_TO_TASK"
        self.objective_index = 0
        self.completed_task_count = 0
        self.stop_started_at = None
        self.park_entry_reached_at = None
        self.completed = False

        self.objective_points = self.build_objective_points()

        self.pub = self.create_publisher(String, self.mission_topic, 10)
        self.event_pub = self.create_publisher(String, self.event_topic, 10)

        self.create_subscription(NavSatFix, self.gnss_topic, self.gnss_cb, 10)
        self.timer = self.create_timer(0.2, self.tick)

        self.get_logger().info(
            "TEKNOFEST mission loaded: "
            + json.dumps(mission_to_dict(self.mission), ensure_ascii=False)
        )

        self.get_logger().info(
            "MISSION_OBJECTIVE_SEQUENCE "
            + json.dumps(
                [
                    {
                        "index": i,
                        "name": p.name,
                        "kind": self.point_kind(p),
                        "nokta_id": p.nokta_id,
                        "carla_x": p.carla_x,
                        "carla_y": p.carla_y,
                        "competition_mode": self.competition_mode,
                    }
                    for i, p in enumerate(self.objective_points)
                ],
                ensure_ascii=False,
            )
        )

    def connect_to_carla(self):
        try:
            self.carla = load_carla(self.carla_root)
            self.client = self.carla.Client(self.host, self.port)
            self.client.set_timeout(self.timeout)
            self.world = self.client.get_world()
            self.get_logger().info(
                f"Mission node CARLA XY distance aktif: "
                f"{self.host}:{self.port} map={self.world.get_map().name}"
            )
        except Exception as exc:
            self.get_logger().warning(
                f"Mission node CARLA bağlantısı kurulamadı, GNSS fallback kullanılacak: {exc}"
            )
            self.use_carla_xy_distance = False

    def find_ego(self):
        if self.world is None:
            return None

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
                    self.get_logger().info(f"Mission node ego bulundu: id={vehicle.id}")
                    return self.ego

        except Exception as exc:
            self.get_logger().warning(f"Mission node ego arama hatası: {exc}")

        return None

    def current_carla_location(self):
        ego = self.find_ego()

        if ego is None:
            return None

        try:
            return ego.get_location()
        except Exception:
            return None

    def current_carla_transform(self):
        ego = self.find_ego()

        if ego is None:
            return None

        try:
            return ego.get_transform()
        except Exception:
            return None

    def current_speed_mps(self):
        ego = self.find_ego()
        if ego is None:
            return None

        try:
            velocity = ego.get_velocity()
            return math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)
        except Exception:
            return None

    def front_bumper_offset(self):
        if self.front_bumper_offset_m > 0.0:
            return self.front_bumper_offset_m

        ego = self.find_ego()
        if ego is not None:
            try:
                return float(ego.bounding_box.extent.x)
            except Exception:
                pass

        return 1.35

    def current_carla_reference_location(self):
        transform = self.current_carla_transform()
        if transform is None:
            return None

        if self.distance_reference != "front_bumper":
            self.last_distance_reference_used = "vehicle_origin"
            return transform.location

        yaw = math.radians(float(transform.rotation.yaw))
        offset = self.front_bumper_offset()
        self.last_distance_reference_used = "front_bumper"

        return self.carla.Location(
            x=float(transform.location.x) + math.cos(yaw) * offset,
            y=float(transform.location.y) + math.sin(yaw) * offset,
            z=float(transform.location.z),
        )

    def publish_event(self, event_type: str, payload: dict):
        msg = String()
        data = {
            "stamp": time.time(),
            "event_type": event_type,
            **payload,
        }
        msg.data = json.dumps(data, ensure_ascii=False)
        self.event_pub.publish(msg)
        self.get_logger().info(f"[MISSION EVENT] {msg.data}")

    def gnss_cb(self, msg: NavSatFix):
        self.current_lat = float(msg.latitude)
        self.current_lon = float(msg.longitude)

    def point_kind(self, point):
        explicit_kind = str(getattr(point, "kind", "") or "").lower()
        name = str(point.name or "").lower()

        if explicit_kind in {"start", "via", "task", "park"}:
            return explicit_kind

        if name == "start":
            return "start"

        if name == "park_giris" or name.startswith("park"):
            return "park"

        if name.startswith("gorev_") or name.startswith("passenger_"):
            return "task"

        if name.startswith("via_"):
            return "via"

        return "unknown"

    def build_objective_points(self):
        if self.competition_mode:
            # Yarışma modu: sadece görev noktaları ve park giriş.
            # Rota/via yok. Planner bu hedeflere göre rotayı kendisi üretecek.
            return list(self.mission.task_points) + [self.mission.park_entry]

        # Eski/debug mod: via dahil sıralı dosyayı yine çalıştırabilir.
        points = [
            p for name, p in self.mission.raw_points.items()
            if str(name).lower() != "start"
        ]

        points = sorted(points, key=lambda p: int(p.nokta_id))

        non_park = [p for p in points if self.point_kind(p) != "park"]
        park = [p for p in points if self.point_kind(p) == "park"]

        route = non_park + park

        if not route:
            route = list(self.mission.task_points) + [self.mission.park_entry]

        return route

    def current_objective_point(self):
        if self.objective_index < len(self.objective_points):
            return self.objective_points[self.objective_index]

        return self.mission.park_entry

    def current_objective_kind(self):
        return self.point_kind(self.current_objective_point())

    def advance_objective_index(self):
        if self.objective_index < len(self.objective_points) - 1:
            self.objective_index += 1
            return True

        return False

    # Eski RouteAgent uyumluluğu için isimleri koruyoruz.
    def target_point(self):
        return self.current_objective_point()

    def current_route_point(self):
        return self.current_objective_point()

    def current_route_kind(self):
        return self.current_objective_kind()

    @property
    def route_index(self):
        return self.objective_index

    def distance_to_target(self):
        target = self.target_point()

        if (
            self.use_carla_xy_distance
            and target.carla_x is not None
            and target.carla_y is not None
        ):
            loc = self.current_carla_reference_location()

            if loc is not None:
                return math.hypot(
                    float(loc.x) - float(target.carla_x),
                    float(loc.y) - float(target.carla_y),
                )

        if self.current_lat is None or self.current_lon is None:
            return None

        return haversine_meters(
            self.current_lat,
            self.current_lon,
            target.lat,
            target.lon,
        )

    def start_passenger_stop(self, target, dist, now):
        self.stage = "PASSENGER_STOP"
        self.stop_started_at = now
        self.publish_event(
            "passenger_stop_started",
            {
                "target": asdict(target),
                "task_index": self.completed_task_count,
                "objective_index": self.objective_index,
                "route_index": self.objective_index,
                "distance_m": round(dist, 3),
            },
        )

    def complete_passenger_stop(self, target, elapsed):
        self.publish_event(
            "passenger_stop_completed",
            {
                "target": asdict(target),
                "task_index": self.completed_task_count,
                "objective_index": self.objective_index,
                "route_index": self.objective_index,
                "stop_elapsed_s": round(elapsed, 3),
                "valid_stop_window": elapsed <= self.passenger_stop_max_s,
            },
        )

        self.completed_task_count += 1
        self.stop_started_at = None

        self.advance_objective_index()

        next_kind = self.current_objective_kind()
        if next_kind == "park":
            self.stage = "GO_TO_PARK"
        else:
            self.stage = "GO_TO_TASK"

        next_target = self.current_objective_point()
        self.publish_event(
            "mission_released_to_next_objective",
            {
                "next_target": asdict(next_target),
                "next_objective_index": self.objective_index,
                "next_stage": self.stage,
                "must_stop": False,
            },
        )

    def start_parking_stage(self, target, dist, now):
        self.stage = "PARKING"
        self.park_entry_reached_at = now
        self.publish_event(
            "park_entry_reached",
            {
                "target": asdict(target),
                "objective_index": self.objective_index,
                "route_index": self.objective_index,
                "distance_m": round(dist, 3),
            },
        )

    def tick(self):
        dist = self.distance_to_target()
        target = self.target_point()
        kind = self.current_objective_kind()
        now = time.time()

        if dist is not None and not self.completed:
            if self.stage in {"GO_TO_TASK", "GO_TO_PARK"}:
                if kind == "via" and not self.competition_mode and dist <= self.point_pass_tolerance_m:
                    self.publish_event(
                        "route_via_reached",
                        {
                            "target": asdict(target),
                            "objective_index": self.objective_index,
                            "route_index": self.objective_index,
                            "distance_m": round(dist, 3),
                        },
                    )

                    self.advance_objective_index()
                    kind = self.current_objective_kind()

                    if kind == "park":
                        self.stage = "GO_TO_PARK"
                    else:
                        self.stage = "GO_TO_TASK"

                elif kind == "task" and dist <= self.task_stop_tolerance_m:
                    self.start_passenger_stop(target, dist, now)

                elif kind == "park" and dist <= self.park_entry_tolerance_m:
                    self.start_parking_stage(target, dist, now)

            elif self.stage == "PASSENGER_STOP":
                elapsed = now - self.stop_started_at

                if elapsed >= self.passenger_stop_min_s:
                    self.complete_passenger_stop(target, elapsed)

            elif self.stage == "PARKING":
                elapsed = now - self.park_entry_reached_at

                if elapsed > self.park_time_limit_s:
                    self.completed = True
                    self.stage = "FAILED"
                    self.publish_event(
                        "park_timeout",
                        {
                            "park_elapsed_s": round(elapsed, 3),
                            "park_time_limit_s": self.park_time_limit_s,
                        },
                    )

                elif elapsed >= self.temporary_parking_complete_s:
                    self.completed = True
                    self.stage = "COMPLETED"
                    self.publish_event(
                        "mission_completed",
                        {
                            "park_elapsed_s": round(elapsed, 3),
                            "within_park_time_limit": elapsed <= self.park_time_limit_s,
                            "temporary_completion": True,
                            "note": "Parking planner eklenene kadar geçici tamamlanma.",
                        },
                    )

        target = self.target_point()
        kind = self.current_objective_kind()

        out = {
            "stamp": now,
            "competition_mode": self.competition_mode,
            "mission": mission_to_dict(self.mission),

            # Eski RouteAgent uyumluluğu.
            "stage": self.stage,
            "task_index": self.completed_task_count,
            "route_index": self.objective_index,
            "route_kind": kind,
            "target": asdict(target),
            "distance_to_target_m": round(dist, 3) if dist is not None else None,
            "distance_reference": self.last_distance_reference_used,
            "task_stop_tolerance_m": self.task_stop_tolerance_m,
            "front_bumper_offset_m": round(self.front_bumper_offset(), 3)
            if self.use_carla_xy_distance else None,

            # Yeni planner uyumluluğu.
            "objective_index": self.objective_index,
            "objective_kind": kind,
            "objective_target": asdict(target),
            "distance_to_objective_m": round(dist, 3) if dist is not None else None,

            "must_stop": self.stage in {"PASSENGER_STOP", "PARKING"},
            "completed": self.completed,
            "passenger_stop_elapsed_s": round(now - self.stop_started_at, 3)
            if self.stop_started_at is not None else None,
            "park_elapsed_s": round(now - self.park_entry_reached_at, 3)
            if self.park_entry_reached_at is not None else None,
        }

        msg = String()
        msg.data = json.dumps(out, ensure_ascii=False)
        self.pub.publish(msg)
        self.log_runtime(out)

    def log_runtime(self, payload: dict):
        target = payload.get("objective_target") or {}
        loc = self.current_carla_reference_location()
        transform = self.current_carla_transform()
        speed_mps = self.current_speed_mps()
        record = {
            "ego_x": round(float(loc.x), 4) if loc is not None else None,
            "ego_y": round(float(loc.y), 4) if loc is not None else None,
            "ego_yaw": round(float(transform.rotation.yaw), 4) if transform is not None else None,
            "current_speed_mps": round(float(speed_mps), 4) if speed_mps is not None else None,
            "active_mission_target": target.get("name"),
            "mission_state": payload.get("stage"),
            "route_index": payload.get("objective_index"),
            "distance_to_goal_m": payload.get("distance_to_objective_m"),
            "distance_reference": payload.get("distance_reference"),
            "task_stop_tolerance_m": payload.get("task_stop_tolerance_m"),
            "front_bumper_offset_m": payload.get("front_bumper_offset_m"),
            "must_stop": payload.get("must_stop"),
            "passenger_stop_elapsed_s": payload.get("passenger_stop_elapsed_s"),
            "completed": payload.get("completed"),
        }
        self.runtime_logger.write(record)

        now = time.time()
        if now - self.last_ros_log_s >= self.ros_log_period_s:
            self.last_ros_log_s = now
            self.get_logger().info(
                "mission_runtime "
                f"state={record['mission_state']} target={record['active_mission_target']} "
                f"idx={record['route_index']} dist={record['distance_to_goal_m']} "
                f"ref={record['distance_reference']} must_stop={record['must_stop']} "
                f"elapsed={record['passenger_stop_elapsed_s']}"
            )


def main(args=None):
    rclpy.init(args=args)
    node = TeknofestMissionNode()

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
