import json
import math
import os
import sys
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from teknofest_sim.carla_loader import load_carla
from teknofest_sim.sign_semantics import (
    constraint_blocks_maneuver,
    road_option_to_maneuver,
    summarize_constraints_for_log,
)
from teknofest_sim.route_constraint_manager import RouteConstraintManager
from teknofest_sim.stopline_manager import StoplineManager


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
        self.declare_parameter("route_constraints_topic", "/adas/perception/route_constraints_json")
        self.declare_parameter("route_intent_topic", "/adas/planning/route_intent")
        self.declare_parameter("route_debug_topic_v2", "/adas/route/debug")
        self.declare_parameter("route_intent_topic_v2", "/adas/route/intent")
        self.declare_parameter("route_constraint_timeout_s", 2.5)
        self.declare_parameter("route_constraint_block_on_violation", False)
        self.declare_parameter("manual_tl_stoplines_enabled", True)
        self.declare_parameter("manual_tl_stoplines_path", "autonomous_driving/configs/manual_stoplines_town03.json")
        self.declare_parameter("front_bumper_offset_m", -1.0)
        self.declare_parameter("default_stop_before_line_m", 1.0)

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
        self.route_constraints_topic = str(self.get_parameter("route_constraints_topic").value)
        self.route_intent_topic = str(self.get_parameter("route_intent_topic").value)
        self.route_debug_topic_v2 = str(self.get_parameter("route_debug_topic_v2").value)
        self.route_intent_topic_v2 = str(self.get_parameter("route_intent_topic_v2").value)
        self.route_constraint_timeout_s = float(self.get_parameter("route_constraint_timeout_s").value)
        self.route_constraint_block_on_violation = bool(self.get_parameter("route_constraint_block_on_violation").value)
        self.manual_tl_stoplines_enabled = bool(self.get_parameter("manual_tl_stoplines_enabled").value)
        self.manual_tl_stoplines_path = str(self.get_parameter("manual_tl_stoplines_path").value)
        self.front_bumper_offset_m = float(self.get_parameter("front_bumper_offset_m").value)
        self.default_stop_before_line_m = float(self.get_parameter("default_stop_before_line_m").value)

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
        self.manual_tl_stoplines = self.load_manual_tl_stoplines()
        self.stopline_manager = StoplineManager(
            self.manual_tl_stoplines,
            default_stop_before_m=self.default_stop_before_line_m,
        )
        self.route_constraint_manager = RouteConstraintManager(
            timeout_s=self.route_constraint_timeout_s,
        )

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
        self.latest_route_constraints = []
        self.last_route_constraints_time = 0.0
        self.last_route_intent_payload = None

        self.global_route_pub = self.create_publisher(String, self.global_route_topic, 10)
        self.local_target_pub = self.create_publisher(String, self.local_target_topic, 10)
        self.route_intent_pub = self.create_publisher(String, self.route_intent_topic, 10)
        self.route_intent_pub_v2 = self.create_publisher(String, self.route_intent_topic_v2, 10)
        self.debug_pub = self.create_publisher(String, self.debug_topic, 10)
        self.debug_pub_v2 = self.create_publisher(String, self.route_debug_topic_v2, 10)

        self.create_subscription(String, self.mission_topic, self.mission_cb, 10)
        self.create_subscription(String, self.route_constraints_topic, self.route_constraints_cb, 10)

        period = 1.0 / max(1.0, self.local_target_rate_hz)
        self.timer = self.create_timer(period, self.tick)

        self.get_logger().info(
            f"GlobalRoutePlannerNode hazır: map={self.map.name} "
            f"mission={self.mission_topic} global={self.global_route_topic} "
            f"local={self.local_target_topic} route_constraints={self.route_constraints_topic} route_intent={self.route_intent_topic}"
        )

    def load_manual_tl_stoplines(self):
        if not self.manual_tl_stoplines_enabled:
            return []

        path = os.path.expanduser(str(self.manual_tl_stoplines_path or "").strip())
        if path and not os.path.isabs(path):
            path = os.path.join(os.getcwd(), path)

        if not path or not os.path.exists(path):
            self.get_logger().warning(f"Planner manual stoplines disabled, file missing: {path}")
            return []

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            items = data.get("stoplines", data if isinstance(data, list) else [])
            out = []
            for item in items:
                try:
                    out.append({
                        "id": str(item.get("id", f"stopline_{len(out) + 1}")),
                        "map": str(item.get("map", "")),
                        "x": float(item["x"]),
                        "y": float(item["y"]),
                        "z": float(item.get("z", 0.0) or 0.0),
                        "yaw_deg": float(item.get("yaw_deg", 0.0)),
                        "approach_m": float(item.get("approach_m", 80.0)),
                        "release_after_pass_m": float(item.get("release_after_pass_m", 4.0)),
                        "lateral_half_width_m": float(item.get("lateral_half_width_m", 6.0)),
                        "stop_before_m": float(item.get("stop_before_m", item.get("stop_margin_m", self.default_stop_before_line_m))),
                        "road_id": item.get("road_id"),
                        "lane_id": item.get("lane_id"),
                    })
                except Exception as exc:
                    self.get_logger().warning(f"Planner stopline skipped: {exc} item={item}")
            self.get_logger().info(f"Planner manual stoplines loaded: {len(out)} from {path}")
            return out
        except Exception as exc:
            self.get_logger().warning(f"Planner manual stoplines load failed: {exc}")
            return []

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

    def route_constraints_cb(self, msg):
        try:
            data = json.loads(msg.data)
            constraints = data.get("constraints", [])
            if not isinstance(constraints, list):
                constraints = []

            self.latest_route_constraints = constraints
            self.last_route_constraints_time = time.time()

            if constraints:
                self.get_logger().info(
                    "ROUTE_CONSTRAINT_RX "
                    + json.dumps(summarize_constraints_for_log(constraints[:5]), ensure_ascii=False),
                    throttle_duration_sec=0.5,
                )

        except Exception as exc:
            self.get_logger().warning(f"Route constraint parse hatası: {exc}")

    def fresh_route_constraints(self):
        return self.route_constraint_manager.filter_constraints(
            self.latest_route_constraints,
            time.time(),
            self.last_route_constraints_time,
        )

    def current_speed_mps(self):
        ego = self.find_ego()
        if ego is None:
            return 0.0
        try:
            v = ego.get_velocity()
            return math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)
        except Exception:
            return 0.0

    def ego_front_context(self):
        ego = self.find_ego()
        if ego is None:
            return None
        try:
            tf = ego.get_transform()
            loc = tf.location
            yaw = math.radians(float(tf.rotation.yaw))
            extent_x = self.front_bumper_offset_m
            if extent_x <= 0.0:
                try:
                    extent_x = float(ego.bounding_box.extent.x)
                except Exception:
                    extent_x = 2.25
            return {
                "x": float(loc.x) + math.cos(yaw) * extent_x,
                "y": float(loc.y) + math.sin(yaw) * extent_x,
                "yaw_deg": float(tf.rotation.yaw),
                "fwd": tf.get_forward_vector(),
                "right": tf.get_right_vector(),
                "road_id": getattr(self.map.get_waypoint(loc, project_to_road=True, lane_type=self.carla.LaneType.Driving), "road_id", None),
                "lane_id": getattr(self.map.get_waypoint(loc, project_to_road=True, lane_type=self.carla.LaneType.Driving), "lane_id", None),
            }
        except Exception:
            return None

    def empty_stopline_context(self, reason):
        return {
            "distance_to_stopline_m": None,
            "stopline_id": None,
            "stopline_valid": False,
            "stopline_pose": None,
            "stopline_lateral_m": None,
            "stop_before_m": round(float(self.default_stop_before_line_m), 3),
            "stop_target_dist_m": None,
            "release_after_pass_m": 2.0,
            "stopline_road_id": None,
            "stopline_lane_id": None,
            "road_match": False,
            "lane_match": False,
            "route_connected_match": False,
            "route_valid": False,
            "route_valid_reason": None,
            "yaw_diff_deg": None,
            "selection_reject_reason": reason,
            "debug": reason,
        }

    def nearest_stopline_context(self, local=None, nearest=None):
        if not self.manual_tl_stoplines:
            return self.empty_stopline_context("no_manual_stoplines")

        ctx = self.ego_front_context()
        if ctx is None:
            return self.empty_stopline_context("ego_missing")

        try:
            map_name = str(self.map.name).split("/")[-1]
        except Exception:
            map_name = ""

        route_lane_id = local.get("lane_id") if isinstance(local, dict) else None
        route_road_id = local.get("road_id") if isinstance(local, dict) else None
        if route_lane_id is None and isinstance(nearest, dict):
            route_lane_id = nearest.get("lane_id")
        if route_road_id is None and isinstance(nearest, dict):
            route_road_id = nearest.get("road_id")

        selected = self.stopline_manager.select(
            ctx,
            map_name=map_name,
            route_lane_id=route_lane_id,
            route_road_id=route_road_id,
        )
        selected["stopline_road_id"] = selected.get("road_id")
        selected["stopline_lane_id"] = selected.get("lane_id")
        return selected

    def allowed_maneuvers_from_constraints(self, maneuver, constraints):
        universe = {"left", "right", "straight", "lane_keep"}
        allowed = set(universe)
        forbidden = set()
        for constraint in constraints:
            for item in constraint.get("forbidden_maneuvers", []) or []:
                item = str(item)
                forbidden.add(item)
                allowed.discard(item)

            required = {str(x) for x in constraint.get("required_maneuvers", []) or []}
            if required:
                allowed &= required

            explicit_allowed = {str(x) for x in constraint.get("allowed_maneuvers", []) or []}
            if explicit_allowed:
                allowed &= explicit_allowed

        if maneuver == "straight":
            allowed.add("lane_keep")
        return sorted(allowed), sorted(forbidden)

    def distance_to_next_turn(self, ego_s):
        for sample in self.route_samples:
            sample_s = float(sample.get("distance_m", 0.0))
            if sample_s <= ego_s:
                continue
            maneuver = road_option_to_maneuver(sample.get("road_option"))
            if maneuver in {"left", "right"}:
                return round(sample_s - float(ego_s), 3)
        return None

    def build_route_intent_payload(self, local, nearest, ego_s, target_s, remaining_m, lateral_error_m):
        road_option = str(local.get("road_option", "UNKNOWN"))
        maneuver = road_option_to_maneuver(road_option)
        constraints = self.fresh_route_constraints()
        constraint_eval = self.route_constraint_manager.evaluate(maneuver, constraints)
        allowed_maneuvers = constraint_eval["allowed_maneuvers"]
        forbidden_maneuvers = constraint_eval["forbidden_maneuvers"]
        preferred_side = constraint_eval["preferred_side"]
        route_decision_status = constraint_eval["status"]
        route_decision_reason = constraint_eval["reason"]
        blocking_constraints = constraint_eval["blocking_constraints"]
        non_blocking_constraints = constraint_eval["non_blocking_constraints"]
        stopline = self.nearest_stopline_context(local=local, nearest=nearest)

        payload = {
            "stamp": time.time(),
            "source": "global_route_planner_node",
            "ttl_sec": 1.0,
            "schema_version": "route_intent_v1",
            "route_id": self.route_id,
            "target_name": self.active_target.get("name") if isinstance(self.active_target, dict) else None,
            "mission_stage": self.latest_mission.get("stage") if self.latest_mission else None,
            "objective_index": self.latest_mission.get("objective_index", self.latest_mission.get("route_index"))
            if self.latest_mission else None,
            "objective_kind": self.latest_mission.get("objective_kind", self.latest_mission.get("route_kind"))
            if self.latest_mission else None,

            "route_status": self.route_status,
            "route_decision_status": route_decision_status,
            "route_decision_reason": route_decision_reason,

            "current_road_option": road_option,
            "current_maneuver": maneuver,
            "next_maneuver": "lane_keep" if maneuver == "straight" else maneuver,
            "route_intent": maneuver,
            "desired_maneuver": maneuver,
            "target_lane": local.get("lane_id"),
            "preferred_side": preferred_side,
            "preferred_lane_side": preferred_side,
            "target_lane_id": local.get("lane_id"),
            "current_lane_id": nearest.get("lane_id") if isinstance(nearest, dict) else None,
            "desired_lane_id": local.get("lane_id"),
            "lane_source": "route_sample_lane_center_right_preferred",
            "allowed_maneuvers": allowed_maneuvers,
            "forbidden_maneuvers": forbidden_maneuvers,
            "local_target": {
                "x": local.get("x"),
                "y": local.get("y"),
                "z": local.get("z"),
                "yaw": local.get("yaw"),
                "road_id": local.get("road_id"),
                "lane_id": local.get("lane_id"),
                "distance_m": local.get("distance_m"),
            },
            "nearest_route": {
                "idx": nearest.get("idx") if isinstance(nearest, dict) else None,
                "road_option": nearest.get("road_option") if isinstance(nearest, dict) else None,
                "distance_m": nearest.get("distance_m") if isinstance(nearest, dict) else None,
            },

            "ego_s_m": round(float(ego_s), 3),
            "target_s_m": round(float(target_s), 3),
            "remaining_m": round(float(remaining_m), 3),
            "lateral_error_m": round(float(lateral_error_m), 3),
            "distance_to_next_turn": self.distance_to_next_turn(float(ego_s)),
            "distance_to_task_target": round(float(remaining_m), 3),
            "desired_heading": local.get("yaw"),
            "route_confidence": 1.0 if self.route_status == "route_ready" else 0.0,
            "current_speed_mps": round(float(self.current_speed_mps()), 3),
            "lane_keep_required": True,
            "lane_keep_status": {
                "preferred_side": preferred_side,
                "target_lane_id": local.get("lane_id"),
                "current_lane_id": nearest.get("lane_id") if isinstance(nearest, dict) else None,
                "desired_lane_id": local.get("lane_id"),
                "lane_source": "route_sample_lane_center_right_preferred",
                "lateral_error_m": round(float(lateral_error_m), 3),
                "reason": "RIGHT_LANE_PREFERRED:right_hand_traffic_default"
                if preferred_side == "right" else "lane_side_constraint_preference",
            },
            "distance_to_stopline_m": stopline["distance_to_stopline_m"],
            "stopline_id": stopline["stopline_id"],
            "stopline_valid": stopline["stopline_valid"],
            "stopline_pose": stopline["stopline_pose"],
            "stopline_lateral_m": stopline["stopline_lateral_m"],
            "stop_before_m": stopline["stop_before_m"],
            "stop_target_dist_m": stopline["stop_target_dist_m"],
            "release_after_pass_m": stopline.get("release_after_pass_m"),
            "stopline_road_id": stopline.get("stopline_road_id"),
            "stopline_lane_id": stopline.get("stopline_lane_id"),
            "stopline_route_valid": bool(stopline.get("route_valid", False)),
            "stopline_road_match": bool(stopline.get("road_match", False)),
            "stopline_lane_match": bool(stopline.get("lane_match", False)),
            "stopline_route_connected_match": bool(stopline.get("route_connected_match", False)),
            "stopline_route_valid_reason": stopline.get("route_valid_reason"),
            "stopline_yaw_diff_deg": stopline.get("yaw_diff_deg"),
            "stopline_selection_reject_reason": stopline.get("selection_reject_reason"),
            "stopline_context": stopline,
            "route_constraint_status": {
                "status": route_decision_status,
                "reason": route_decision_reason,
                "allowed_maneuvers": allowed_maneuvers,
                "forbidden_maneuvers": forbidden_maneuvers,
                "preferred_side": preferred_side,
            },

            "active_constraints": summarize_constraints_for_log(constraints),
            "active_route_constraints": summarize_constraints_for_log(constraints),
            "blocking_constraints": blocking_constraints,
            "non_blocking_constraints": non_blocking_constraints,
            "constraint_age_s": round(time.time() - self.last_route_constraints_time, 3)
            if self.last_route_constraints_time else None,
            "debug_reason": route_decision_reason,
            "debug": {
                "stopline": stopline["debug"],
                "right_hand_traffic": True,
            },
        }

        return payload

    def publish_route_intent(self, route_intent):
        msg = String()
        msg.data = json.dumps(route_intent, ensure_ascii=False)
        self.route_intent_pub.publish(msg)
        self.route_intent_pub_v2.publish(msg)
        self.last_route_intent_payload = route_intent

        if route_intent.get("route_decision_status") != "clear":
            marker = (
                "ROUTE_CONSTRAINT_BLOCKED"
                if route_intent.get("route_decision_status") == "blocked"
                else "ROUTE_CONSTRAINT_APPLIED"
            )
            self.get_logger().info(
                f"{marker} ROUTE_INTENT "
                f"status={route_intent.get('route_decision_status')} "
                f"intent={route_intent.get('route_intent')} "
                f"reason={route_intent.get('route_decision_reason')} "
                f"constraints={[x.get('sign_type') for x in route_intent.get('active_constraints', [])]}",
                throttle_duration_sec=0.5,
            )
        stopline_debug = str(route_intent.get("debug", {}).get("stopline", ""))
        if stopline_debug.startswith("TL_STOPLINE"):
            self.get_logger().info(
                stopline_debug,
                throttle_duration_sec=0.5,
            )

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

        route_intent = self.build_route_intent_payload(
            local=local,
            nearest=nearest,
            ego_s=ego_s,
            target_s=target_s,
            remaining_m=remaining_m,
            lateral_error_m=lateral_error_m,
        )
        if route_intent.get("route_decision_status") == "blocked":
            speed_hint = 0.0
            if self.route_constraint_block_on_violation:
                # If configured to block on route-constraint violations,
                # mark the planner route status so downstream nodes can observe it.
                self.route_status = "blocked"

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
            "route_intent": route_intent.get("route_intent"),
            "route_decision_status": route_intent.get("route_decision_status"),
            "route_decision_reason": route_intent.get("route_decision_reason"),
            "preferred_lane_side": route_intent.get("preferred_lane_side"),
            "stopline_context": route_intent.get("stopline_context"),
            "route_constraint_status": route_intent.get("route_constraint_status"),
            "active_route_constraints": route_intent.get("active_constraints", []),
            "blocking_route_constraints": route_intent.get("blocking_constraints", []),
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

        self.publish_route_intent(route_intent)

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
            "last_route_intent": self.last_route_intent_payload,
            "fresh_route_constraints": summarize_constraints_for_log(self.fresh_route_constraints()),
        }

        msg = String()
        msg.data = json.dumps(payload, ensure_ascii=False)
        self.debug_pub.publish(msg)
        self.debug_pub_v2.publish(msg)

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
