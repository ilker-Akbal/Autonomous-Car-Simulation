import json
import math
import os
import sys
import time
from dataclasses import dataclass

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from teknofest_sim.carla_loader import load_carla


@dataclass
class MissionPoint:
    name: str
    nokta_id: int
    x: float
    y: float
    z: float
    yaw: float = 0.0


@dataclass
class RouteSample:
    waypoint: object
    road_option: object
    distance_m: float


@dataclass
class SignPlanItem:
    sign_name: str
    distance_m: float
    side: str
    reason: str
    route_x: object = None
    route_y: object = None
    route_z: object = None
    route_yaw: object = None
    route_index: object = None
    source_event: str = ""


class TeknofestRouteSignsNode(Node):
    def __init__(self):
        super().__init__("teknofest_route_signs_node")

        self.declare_parameter("carla_root", "/home/ilker/simulators/CARLA_0.9.15_SOURCE")
        self.declare_parameter("host", "127.0.0.1")
        self.declare_parameter("port", 2000)
        self.declare_parameter("timeout", 120.0)
        self.declare_parameter("mission_geojson", "")

        self.declare_parameter("enabled", True)
        self.declare_parameter("clear_existing_signs", True)
        self.declare_parameter("destroy_on_shutdown", True)

        self.declare_parameter("route_sampling_resolution_m", 2.0)

        # Eski sabit offset mantığı yerine artık dış şerit + kaldırım payı kullanılıyor.
        self.declare_parameter("curb_extra_m", 0.85)

        self.declare_parameter("z_offset_m", 0.05)
        self.declare_parameter("spawn_lift_m", 0.20)
        self.declare_parameter("min_sign_spacing_m", 18.0)

        # Blender FBX ekseni için düzeltme.
        self.declare_parameter("mesh_yaw_offset_deg", -90.0)

        self.declare_parameter("auto_turn_signs_enabled", True)
        self.declare_parameter("auto_base_signs_enabled", True)

        # Boş bırakılırsa rota analizinden otomatik plan çıkarılır.
        # Manuel örnek:
        # "35:hiz_siniri_30:R,85:yaya_gecidi:R,145:yol_ver:R"
        self.declare_parameter("sign_plan", "")
        self.declare_parameter("sign_plan_file", "")

        self.declare_parameter("debug_draw", False)
        self.declare_parameter("status_topic", "/adas/teknofest/route_signs_status")

        self.carla_root = str(self.get_parameter("carla_root").value)
        self.host = str(self.get_parameter("host").value)
        self.port = int(self.get_parameter("port").value)
        self.timeout = float(self.get_parameter("timeout").value)
        self.mission_geojson = str(self.get_parameter("mission_geojson").value)

        self.enabled = bool(self.get_parameter("enabled").value)
        self.clear_existing_signs = bool(self.get_parameter("clear_existing_signs").value)
        self.destroy_on_shutdown = bool(self.get_parameter("destroy_on_shutdown").value)

        self.route_sampling_resolution_m = float(self.get_parameter("route_sampling_resolution_m").value)
        self.curb_extra_m = float(self.get_parameter("curb_extra_m").value)

        self.z_offset_m = float(self.get_parameter("z_offset_m").value)
        self.spawn_lift_m = float(self.get_parameter("spawn_lift_m").value)
        self.min_sign_spacing_m = float(self.get_parameter("min_sign_spacing_m").value)
        self.mesh_yaw_offset_deg = float(self.get_parameter("mesh_yaw_offset_deg").value)

        self.auto_turn_signs_enabled = bool(self.get_parameter("auto_turn_signs_enabled").value)
        self.auto_base_signs_enabled = bool(self.get_parameter("auto_base_signs_enabled").value)
        self.sign_plan_text = str(self.get_parameter("sign_plan").value)
        self.sign_plan_file = str(self.get_parameter("sign_plan_file").value).strip()

        self.debug_draw = bool(self.get_parameter("debug_draw").value)
        self.status_topic = str(self.get_parameter("status_topic").value)

        self.carla = load_carla(self.carla_root)
        self.client = self.carla.Client(self.host, self.port)
        self.client.set_timeout(self.timeout)
        self.world = self.client.get_world()
        self.map = self.world.get_map()
        self.bp_lib = self.world.get_blueprint_library()

        self.created_actors = []
        self.spawn_reports = []

        self.status_pub = self.create_publisher(String, self.status_topic, 10)

        if self.enabled:
            self.setup_route_signs()
        else:
            self.get_logger().info("Route sign spawner disabled.")

        self.timer = self.create_timer(1.0, self.publish_status)

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
        self.add_python_api_paths()

        try:
            from agents.navigation.global_route_planner import GlobalRoutePlanner

            try:
                return GlobalRoutePlanner(self.map, self.route_sampling_resolution_m)
            except TypeError:
                from agents.navigation.global_route_planner_dao import GlobalRoutePlannerDAO

                dao = GlobalRoutePlannerDAO(self.map, self.route_sampling_resolution_m)
                planner = GlobalRoutePlanner(dao)
                planner.setup()
                return planner

        except Exception as exc:
            raise RuntimeError(f"GlobalRoutePlanner import/kurulum hatası: {exc}")

    def load_mission_points(self):
        if not self.mission_geojson:
            raise RuntimeError("mission_geojson boş. Tabela rotası üretilemez.")

        with open(os.path.expanduser(self.mission_geojson), "r", encoding="utf-8") as f:
            data = json.load(f)

        raw_points = []

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

            try:
                yaw = float(props.get("carla_yaw", props.get("yaw", 0.0)))
            except Exception:
                yaw = 0.0

            raw_points.append(MissionPoint(name, nokta_id, float(x), float(y), float(z), yaw))

        start_list = [p for p in raw_points if p.name.lower() == "start"]
        if not start_list:
            raise RuntimeError("Mission dosyasında start noktası yok.")

        start = start_list[0]
        others = [p for p in raw_points if p.name.lower() != "start"]

        non_park = sorted(
            [p for p in others if not p.name.lower().startswith("park")],
            key=lambda p: p.nokta_id,
        )
        park = sorted(
            [p for p in others if p.name.lower().startswith("park")],
            key=lambda p: p.nokta_id,
        )

        ordered = [start] + non_park + park

        self.get_logger().info(
            "ROUTE_SIGN_MISSION_ORDER "
            + json.dumps(
                [
                    {
                        "name": p.name,
                        "nokta_id": p.nokta_id,
                        "x": round(p.x, 2),
                        "y": round(p.y, 2),
                        "z": round(p.z, 2),
                    }
                    for p in ordered
                ],
                ensure_ascii=False,
            )
        )

        return ordered

    def to_location(self, point):
        return self.carla.Location(x=point.x, y=point.y, z=point.z)

    def build_route_samples(self, mission_points):
        planner = self.make_global_route_planner()
        raw_route = []

        for i in range(len(mission_points) - 1):
            start = self.to_location(mission_points[i])
            end = self.to_location(mission_points[i + 1])
            segment = planner.trace_route(start, end)

            if not segment:
                self.get_logger().warning(
                    f"Route segment boş: {mission_points[i].name} -> {mission_points[i + 1].name}"
                )
                continue

            if raw_route:
                segment = segment[1:]

            raw_route.extend(segment)

        if len(raw_route) < 2:
            raise RuntimeError("GlobalRoutePlanner yeterli rota üretemedi.")

        samples = []
        total = 0.0
        prev = None

        for wp, road_option in raw_route:
            loc = wp.transform.location
            if prev is not None:
                total += math.hypot(loc.x - prev.x, loc.y - prev.y)

            samples.append(RouteSample(wp, road_option, total))
            prev = loc

        self.get_logger().info(
            f"ROUTE_SIGN_ROUTE_READY samples={len(samples)} length_m={total:.1f}"
        )
        return samples

    def route_len(self, route_samples):
        return float(route_samples[-1].distance_m) if route_samples else 0.0

    def sample_at_distance(self, route_samples, distance_m):
        target = max(0.0, min(float(distance_m), self.route_len(route_samples)))
        last = route_samples[0]

        for sample in route_samples:
            if sample.distance_m >= target:
                return sample
            last = sample

        return last

    def option_name(self, road_option):
        if road_option is None:
            return "UNKNOWN"

        name = getattr(road_option, "name", None)
        if name:
            return str(name).upper()

        return str(road_option).split(".")[-1].upper()

    def norm_sign(self, name):
        s = str(name).strip().lower()
        s = s.replace("static.prop.teknofest_sign_", "")
        s = s.replace("teknofest_sign_", "")
        return s

    def norm_side(self, side):
        s = str(side or "R").strip().upper()
        if s in {"L", "LEFT", "SOL"}:
            return "L"
        return "R"

    def parse_manual_plan(self):
        text = self.sign_plan_text.strip()
        if not text:
            return []

        items = []

        for token in text.split(","):
            parts = [p.strip() for p in token.split(":")]
            if len(parts) < 2:
                self.get_logger().warning(f"Geçersiz sign_plan token: {token}")
                continue

            try:
                distance_m = float(parts[0])
                sign_name = self.norm_sign(parts[1])
                side = self.norm_side(parts[2] if len(parts) >= 3 else "R")
            except Exception:
                self.get_logger().warning(f"Geçersiz sign_plan token: {token}")
                continue

            items.append(SignPlanItem(sign_name, distance_m, side, "manual"))

        return items

    def parse_plan_file(self):
        if not self.sign_plan_file:
            return []

        path = os.path.expanduser(self.sign_plan_file)

        if not os.path.exists(path):
            raise RuntimeError(f"sign_plan_file bulunamadı: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        raw_items = data.get("signs", data) if isinstance(data, dict) else data

        if not isinstance(raw_items, list):
            raise RuntimeError("sign_plan_file formatı liste veya {'signs': [...]} olmalı.")

        items = []

        for raw in raw_items:
            if not isinstance(raw, dict):
                continue

            sign_name = self.norm_sign(raw.get("sign", raw.get("sign_name", "")))
            if not sign_name:
                continue

            try:
                distance_m = float(raw["distance_m"])
            except Exception as exc:
                raise RuntimeError(f"sign_plan_file içinde distance_m hatalı: {raw}") from exc

            items.append(
                SignPlanItem(
                    sign_name=sign_name,
                    distance_m=distance_m,
                    side=self.norm_side(raw.get("side", "R")),
                    reason=str(raw.get("reason", "plan_file")),
                    route_x=raw.get("route_x"),
                    route_y=raw.get("route_y"),
                    route_z=raw.get("route_z", raw.get("carla_z", 0.2)),
                    route_yaw=raw.get("route_yaw"),
                    route_index=raw.get("route_index"),
                    source_event=str(raw.get("source_event", "")),
                )
            )

        items = sorted(items, key=lambda x: x.distance_m)

        self.get_logger().info(
            "ROUTE_SIGN_FILE_PLAN "
            + json.dumps(
                [
                    {
                        "sign": x.sign_name,
                        "distance_m": round(x.distance_m, 2),
                        "side": x.side,
                        "route_x": x.route_x,
                        "route_y": x.route_y,
                        "reason": x.reason,
                    }
                    for x in items
                ],
                ensure_ascii=False,
            )
        )

        return items

    def add_plan_item(self, items, sign_name, distance_m, side, reason, route_len):
        d = max(8.0, min(float(distance_m), max(8.0, route_len - 8.0)))

        for _ in range(8):
            if all(abs(d - x.distance_m) >= self.min_sign_spacing_m for x in items):
                break
            d = min(route_len - 8.0, d + self.min_sign_spacing_m * 0.65)

        if any(abs(d - x.distance_m) < self.min_sign_spacing_m for x in items):
            self.get_logger().warning(
                f"Sign spacing yüzünden atlandı: {sign_name} d={d:.1f} reason={reason}"
            )
            return

        items.append(SignPlanItem(self.norm_sign(sign_name), d, self.norm_side(side), reason))

    def build_auto_plan(self, route_samples):
        length = self.route_len(route_samples)
        items = []

        if self.auto_base_signs_enabled:
            self.add_plan_item(items, "hiz_siniri_30", 32.0, "R", "start_speed", length)
            self.add_plan_item(items, "yaya_gecidi", length * 0.22, "R", "urban_crosswalk", length)
            self.add_plan_item(items, "hiz_siniri_20", length * 0.46, "R", "passenger_zone_slow", length)
            self.add_plan_item(items, "dur", length * 0.66, "R", "late_section_stop", length)
            self.add_plan_item(items, "park_etmek_yasaktir", length - 62.0, "L", "wrong_parking_side", length)
            self.add_plan_item(items, "park_yeri", length - 34.0, "R", "parking_entry", length)

        if self.auto_turn_signs_enabled:
            last_turn_d = -9999.0
            yol_ver_added = False

            for sample in route_samples:
                opt = self.option_name(sample.road_option)

                if opt not in {"LEFT", "RIGHT"}:
                    continue

                if sample.distance_m < 35.0 or sample.distance_m > length - 50.0:
                    continue

                if sample.distance_m - last_turn_d < 35.0:
                    continue

                sign_distance = sample.distance_m - 24.0

                if not yol_ver_added and sign_distance > 45.0:
                    self.add_plan_item(
                        items,
                        "yol_ver",
                        sign_distance - 22.0,
                        "R",
                        f"before_{opt.lower()}_junction",
                        length,
                    )
                    yol_ver_added = True

                turn_sign = (
                    "ileriden_sola_mecburi_yon"
                    if opt == "LEFT"
                    else "ileriden_saga_mecburi_yon"
                )

                self.add_plan_item(
                    items,
                    turn_sign,
                    sign_distance,
                    "R",
                    f"auto_turn_{opt.lower()}_{sample.distance_m:.1f}m",
                    length,
                )

                last_turn_d = sample.distance_m

        items = sorted(items, key=lambda x: x.distance_m)

        self.get_logger().info(
            "ROUTE_SIGN_AUTO_PLAN "
            + json.dumps(
                [
                    {
                        "sign": x.sign_name,
                        "distance_m": round(x.distance_m, 1),
                        "side": x.side,
                        "reason": x.reason,
                    }
                    for x in items
                ],
                ensure_ascii=False,
            )
        )

        return items

    def blueprint_for_sign(self, sign_name):
        bp_id = f"static.prop.teknofest_sign_{self.norm_sign(sign_name)}"
        matches = list(self.bp_lib.filter(bp_id))

        if matches:
            return matches[0]

        try:
            return self.bp_lib.find(bp_id)
        except Exception:
            return None

    def clear_existing(self):
        destroyed = 0

        for actor in list(self.world.get_actors()):
            try:
                if str(actor.type_id).startswith("static.prop.teknofest_sign_"):
                    actor.destroy()
                    destroyed += 1
            except Exception:
                pass

        self.get_logger().info(f"ROUTE_SIGN_CLEAR destroyed={destroyed}")

    def same_direction_driving_lane(self, base_wp, candidate_wp):
        if candidate_wp is None:
            return False

        try:
            if candidate_wp.lane_type != self.carla.LaneType.Driving:
                return False
        except Exception:
            return False

        try:
            base_lane_id = int(base_wp.lane_id)
            candidate_lane_id = int(candidate_wp.lane_id)

            # Aynı yöndeki driving lane'ler genelde aynı işaretli lane_id taşır.
            if base_lane_id * candidate_lane_id <= 0:
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
        """
        Yol ortası problemini çözen ana mantık.

        R:
            Aynı yöndeki sağ şeritlere yürür, en dış sağ şeridi bulur.

        L:
            Aynı yöndeki sol şeritlere yürür, en dış sol şeridi bulur.

        Tabela bu dış şeridin merkezinden değil,
        dış şerit kenarından kaldırım tarafına konur.
        """
        side = self.norm_side(side)
        current = wp

        for _ in range(8):
            try:
                candidate = current.get_right_lane() if side == "R" else current.get_left_lane()
            except Exception:
                candidate = None

            if not self.same_direction_driving_lane(current, candidate):
                break

            current = candidate

        return current

    def ground_z_at(self, loc, fallback_z):
        try:
            probe = self.carla.Location(x=loc.x, y=loc.y, z=loc.z + 8.0)

            if hasattr(self.world, "ground_projection"):
                projected = self.world.ground_projection(probe, 20.0)
                if projected is not None:
                    return float(projected.location.z) + self.z_offset_m
        except Exception:
            pass

        return float(fallback_z) + self.z_offset_m

    def yaw_to_face_road(self, sign_loc, road_loc):
        dx = float(road_loc.x - sign_loc.x)
        dy = float(road_loc.y - sign_loc.y)
        yaw = math.degrees(math.atan2(dy, dx))
        return yaw + self.mesh_yaw_offset_deg

    def make_transform(self, sample, side, extra_curb_m=0.0):
        side = self.norm_side(side)

        original_wp = sample.waypoint
        outer_wp = self.outermost_same_direction_lane(original_wp, side)

        outer_tf = outer_wp.transform
        outer_loc = outer_tf.location
        right = outer_tf.get_right_vector()

        side_mul = 1.0 if side == "R" else -1.0

        try:
            lane_width = float(outer_wp.lane_width)
        except Exception:
            lane_width = 3.5

        # Şerit merkezinden değil, en dış şerit kenarından kaldırıma çıkıyoruz.
        lateral_m = (lane_width * 0.5) + self.curb_extra_m + float(extra_curb_m)

        sign_loc = self.carla.Location(
            x=outer_loc.x + right.x * lateral_m * side_mul,
            y=outer_loc.y + right.y * lateral_m * side_mul,
            z=outer_loc.z + self.spawn_lift_m,
        )

        ground_z = self.ground_z_at(sign_loc, outer_loc.z)
        sign_loc.z = ground_z + self.spawn_lift_m

        # Tabela yüzünü yola çevir.
        yaw = self.yaw_to_face_road(sign_loc, outer_loc)

        transform = self.carla.Transform(
            sign_loc,
            self.carla.Rotation(pitch=0.0, yaw=yaw, roll=0.0),
        )

        placement_debug = {
            "original_road_id": getattr(original_wp, "road_id", None),
            "original_lane_id": getattr(original_wp, "lane_id", None),
            "outer_road_id": getattr(outer_wp, "road_id", None),
            "outer_lane_id": getattr(outer_wp, "lane_id", None),
            "lane_width": round(lane_width, 3),
            "lateral_from_outer_center_m": round(lateral_m, 3),
            "curb_extra_m": round(self.curb_extra_m, 3),
        }

        return transform, ground_z, placement_debug

    def snap_bottom_to_ground(self, actor, ground_z):
        try:
            tf = actor.get_transform()
            bb = actor.bounding_box

            bottom_z = tf.location.z + bb.location.z - bb.extent.z
            delta_z = float(ground_z) - float(bottom_z)

            if abs(delta_z) > 0.001:
                actor.set_transform(
                    self.carla.Transform(
                        self.carla.Location(
                            x=tf.location.x,
                            y=tf.location.y,
                            z=tf.location.z + delta_z,
                        ),
                        tf.rotation,
                    )
                )
        except Exception as exc:
            self.get_logger().warning(f"GROUND_SNAP_FAILED actor={actor.id}: {exc}")

    def draw_debug(self, actor, item):
        if not self.debug_draw:
            return

        try:
            color = self.carla.Color(0, 255, 80)
            tf = actor.get_transform()

            self.world.debug.draw_string(
                tf.location + self.carla.Location(z=2.2),
                f"{item.sign_name}@{item.distance_m:.0f}m",
                draw_shadow=True,
                color=color,
                life_time=20.0,
                persistent_lines=False,
            )
        except Exception as exc:
            self.get_logger().warning(f"DEBUG_DRAW_FAILED: {exc}")

    def sample_for_plan_item(self, item, route_samples):
        if item.route_x is None or item.route_y is None:
            return self.sample_at_distance(route_samples, item.distance_m)

        try:
            loc = self.carla.Location(
                x=float(item.route_x),
                y=float(item.route_y),
                z=float(item.route_z or 0.2),
            )

            wp = self.map.get_waypoint(
                loc,
                project_to_road=True,
                lane_type=self.carla.LaneType.Driving,
            )

            if wp is None:
                self.get_logger().warning(
                    f"PLAN_ANCHOR_WAYPOINT_NOT_FOUND sign={item.sign_name} "
                    f"x={item.route_x} y={item.route_y}; distance fallback kullanılacak."
                )
                return self.sample_at_distance(route_samples, item.distance_m)

            closest = min(
                route_samples,
                key=lambda s: math.hypot(
                    s.waypoint.transform.location.x - loc.x,
                    s.waypoint.transform.location.y - loc.y,
                ),
            )

            return RouteSample(wp, closest.road_option, float(item.distance_m))

        except Exception as exc:
            self.get_logger().warning(
                f"PLAN_ANCHOR_PARSE_FAILED sign={item.sign_name}: {exc}; distance fallback kullanılacak."
            )
            return self.sample_at_distance(route_samples, item.distance_m)

    def lateral_distance_from_wp(self, wp, loc):
        wp_loc = wp.transform.location
        right = wp.transform.get_right_vector()

        dx = float(loc.x - wp_loc.x)
        dy = float(loc.y - wp_loc.y)

        return dx * float(right.x) + dy * float(right.y)

    def is_on_driving_lane_exact(self, loc):
        try:
            wp = self.map.get_waypoint(
                loc,
                project_to_road=False,
                lane_type=self.carla.LaneType.Driving,
            )
            return wp is not None
        except Exception:
            return False

    def validate_sign_transform(self, sample, transform):
        original_wp = sample.waypoint

        if bool(getattr(original_wp, "is_junction", False)):
            return False, "anchor_inside_junction"

        lateral = abs(self.lateral_distance_from_wp(original_wp, transform.location))

        if lateral < 2.20:
            return False, f"too_close_to_route_lateral_{lateral:.2f}m"

        if lateral > 7.50:
            return False, f"too_far_from_route_lateral_{lateral:.2f}m"

        if self.is_on_driving_lane_exact(transform.location):
            return False, "candidate_on_driving_lane"

        return True, "ok"

    def spawn_sign(self, item, route_samples):
        bp = self.blueprint_for_sign(item.sign_name)

        if bp is None:
            report = {
                "sign": item.sign_name,
                "distance_m": round(item.distance_m, 2),
                "side": item.side,
                "spawned": False,
                "reason": "blueprint_not_found",
            }
            self.spawn_reports.append(report)
            self.get_logger().warning(f"ROUTE_SIGN_BP_NOT_FOUND sign={item.sign_name}")
            return None

        base_sample = self.sample_for_plan_item(item, route_samples)

        rejected = []

        # Final kullanım kuralı:
        # Negatif offset yok. Tabela çakışırsa tekrar yola değil, kaldırıma/dışarı kaçar.
        extra_curb_candidates = [0.0, 0.35, 0.70, 1.05, 1.40, 1.75, 2.10]

        # Plan L diyorsa önce L denenir, olmazsa R denenir.
        # Plan R diyorsa sadece R; çünkü yarış rota tabelaları öncelikle sağ tarafta olmalı.
        side_candidates = ["L", "R"] if self.norm_side(item.side) == "L" else ["R"]

        for side in side_candidates:
            for extra_curb_m in extra_curb_candidates:
                transform, ground_z, placement_debug = self.make_transform(
                    base_sample,
                    side,
                    extra_curb_m=extra_curb_m,
                )

                ok, reject_reason = self.validate_sign_transform(base_sample, transform)

                if not ok:
                    rejected.append(
                        {
                            "side": side,
                            "extra_curb_m": round(extra_curb_m, 2),
                            "reason": reject_reason,
                        }
                    )
                    continue

                actor = self.world.try_spawn_actor(bp, transform)

                if actor is None:
                    rejected.append(
                        {
                            "side": side,
                            "extra_curb_m": round(extra_curb_m, 2),
                            "reason": "try_spawn_actor_failed",
                        }
                    )
                    continue

                for method_name in ["set_simulate_physics", "set_enable_gravity"]:
                    try:
                        getattr(actor, method_name)(False)
                    except Exception:
                        pass

                self.snap_bottom_to_ground(actor, ground_z)

                try:
                    self.world.wait_for_tick(seconds=0.2)
                except Exception:
                    pass

                self.created_actors.append(actor)
                self.draw_debug(actor, item)

                tf = actor.get_transform()

                report = {
                    "id": actor.id,
                    "type_id": actor.type_id,
                    "sign": item.sign_name,
                    "planned_distance_m": round(item.distance_m, 2),
                    "side": side,
                    "x": round(tf.location.x, 3),
                    "y": round(tf.location.y, 3),
                    "z": round(tf.location.z, 3),
                    "yaw": round(tf.rotation.yaw, 2),
                    "road_option": self.option_name(base_sample.road_option),
                    "spawned": True,
                    "reason": item.reason,
                    "source_event": item.source_event,
                    "route_index": item.route_index,
                    "route_anchor_x": item.route_x,
                    "route_anchor_y": item.route_y,
                    "used_extra_curb_m": round(extra_curb_m, 2),
                    "placement": placement_debug,
                    "rejected_candidate_count_before_success": len(rejected),
                }

                self.spawn_reports.append(report)
                self.get_logger().info("ROUTE_SIGN_SPAWNED " + json.dumps(report, ensure_ascii=False))
                return actor

        report = {
            "sign": item.sign_name,
            "distance_m": round(item.distance_m, 2),
            "side": item.side,
            "spawned": False,
            "reason": "no_safe_spawn_candidate",
            "source_event": item.source_event,
            "route_index": item.route_index,
            "route_anchor_x": item.route_x,
            "route_anchor_y": item.route_y,
            "rejected_candidate_count": len(rejected),
            "rejected_candidates_sample": rejected[:30],
        }

        self.spawn_reports.append(report)
        self.get_logger().error("ROUTE_SIGN_SPAWN_FAILED " + json.dumps(report, ensure_ascii=False))
        return None

    def setup_route_signs(self):
        if self.clear_existing_signs:
            self.clear_existing()

        mission_points = self.load_mission_points()
        route_samples = self.build_route_samples(mission_points)

        file_plan = self.parse_plan_file()
        manual_plan = self.parse_manual_plan()

        if file_plan:
            plan = file_plan
            self.get_logger().info("ROUTE_SIGN_PLAN_SOURCE file")
        elif manual_plan:
            plan = manual_plan
            self.get_logger().info("ROUTE_SIGN_PLAN_SOURCE manual")
        else:
            # Final kullanımda sign_plan_file bekliyoruz.
            # Yine de eski launch'lar tamamen kırılmasın diye fallback duruyor.
            plan = self.build_auto_plan(route_samples)
            self.get_logger().warning("ROUTE_SIGN_PLAN_SOURCE automatic_route_based_fallback")

        if not plan:
            raise RuntimeError("Tabela planı boş. Final kullanımda boş plan kabul edilmez.")

        spawned = 0

        for item in plan:
            if self.spawn_sign(item, route_samples) is not None:
                spawned += 1

        if spawned != len(plan):
            self.get_logger().error(
                f"ROUTE_SIGN_SUMMARY spawned={spawned}/{len(plan)} route_len={self.route_len(route_samples):.1f}m"
            )
        else:
            self.get_logger().info(
                f"ROUTE_SIGN_SUMMARY spawned={spawned}/{len(plan)} route_len={self.route_len(route_samples):.1f}m"
            )

    def publish_status(self):
        msg = String()
        msg.data = json.dumps(
            {
                "stamp": round(time.time(), 3),
                "enabled": self.enabled,
                "created_actor_count": len(self.created_actors),
                "alive_actor_count": len(
                    [a for a in self.created_actors if getattr(a, "is_alive", False)]
                ),
                "spawn_reports": self.spawn_reports,
            },
            ensure_ascii=False,
        )
        self.status_pub.publish(msg)

    def destroy_node(self):
        if self.destroy_on_shutdown:
            for actor in reversed(getattr(self, "created_actors", [])):
                try:
                    if actor.is_alive:
                        actor.destroy()
                except Exception:
                    pass

        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = TeknofestRouteSignsNode()

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