import json
import math
import os
import sys
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from teknofest_sim.carla_loader import load_carla


class TeknofestRouteAgentNode(Node):
    def __init__(self):
        super().__init__("teknofest_route_agent_node")

        self.declare_parameter("carla_root", "/home/ilker/simulators/CARLA_0.9.15")
        self.declare_parameter("host", "127.0.0.1")
        self.declare_parameter("port", 2000)
        self.declare_parameter("timeout", 120.0)
        self.declare_parameter("ego_role_name", "ego_vehicle")

        self.declare_parameter("decision_topic", "/adas/decision")
        self.declare_parameter("mission_topic", "/adas/teknofest/mission")

        # Yarışma modu:
        # Mission node sadece görev hedefini verir.
        # GlobalRoutePlannerNode bu hedefe runtime rota üretir.
        # RouteAgent artık mümkünse /adas/planning/local_target takip eder.
        self.declare_parameter("use_planner_local_target", True)
        self.declare_parameter("local_target_topic", "/adas/planning/local_target")
        self.declare_parameter("planner_fresh_timeout_s", 1.0)
        self.declare_parameter("planner_speed_hint_enabled", True)
        self.declare_parameter("planner_local_target_as_destination", True)
        self.declare_parameter("planner_destination_hold_enabled", True)
        self.declare_parameter("planner_destination_reached_m", 7.0)
        self.declare_parameter("planner_destination_update_min_distance_m", 18.0)
        self.declare_parameter("planner_destination_update_min_interval_s", 2.0)
        self.declare_parameter("planner_target_key_resolution_m", 3.0)
        self.declare_parameter("target_speed_smoothing_enabled", True)
        self.declare_parameter("target_speed_accel_limit_mps2", 1.8)
        self.declare_parameter("target_speed_decel_limit_mps2", 3.0)
        self.declare_parameter("green_release_min_start_speed_mps", 2.0)
        self.declare_parameter("tl_stopline_stop_before_m", 2.0)
        self.declare_parameter("tl_passed_stopline_ignore_s", 4.0)
        self.declare_parameter("tl_front_bumper_offset_m", 2.3)
        self.declare_parameter("tl_desired_front_bumper_stop_before_m", 0.8)

        # MANUAL_TL_STOPLINE_FIX:
        # Trafik ışığı mesafesi bbox/depth ile değil, CARLA dünya koordinatındaki
        # durma çizgisine göre yönetilecek.
        self.declare_parameter("manual_tl_stoplines_enabled", True)
        self.declare_parameter(
            "manual_tl_stoplines_path",
            "autonomous_driving/configs/manual_stoplines_town03.json",
        )

        self.declare_parameter("debug_topic", "/adas/teknofest/route_agent_debug")
        self.declare_parameter("collision_topic", "/adas/events/collision")
        self.declare_parameter("collision_halt_s", 4.0)

        self.declare_parameter("control_rate_hz", 20.0)
        self.declare_parameter("max_speed_mps", 4.0)
        self.declare_parameter("go_speed_mps", 3.2)
        self.declare_parameter("slow_speed_mps", 1.4)
        self.declare_parameter("parking_speed_mps", 0.65)

        # İnsan/launch tarafı km/h. Verilirse m/s değerlerini override eder.
        self.declare_parameter("max_speed_kmh", -1.0)
        self.declare_parameter("go_speed_kmh", -1.0)
        self.declare_parameter("slow_speed_kmh", -1.0)
        self.declare_parameter("parking_speed_kmh", -1.0)

        # BasicAgent direksiyonu kendi üretir. Bunu çok düşük tutarsan araç dönemiyor.
        self.declare_parameter("max_steer", 0.70)
        self.declare_parameter("lane_assist_enabled", True)
        self.declare_parameter("lane_topic", "/adas/lane/assist")
        self.declare_parameter("lane_min_confidence", 0.60)
        self.declare_parameter("lane_fresh_timeout_s", 0.50)
        self.declare_parameter("lane_blend_straight", 0.35)
        self.declare_parameter("lane_blend_turn", 0.12)
        self.declare_parameter("lane_turn_steer_threshold", 0.28)
        self.declare_parameter("lane_allowed_stages", "GO_TO_TASK,GO_TO_PARK")

        self.declare_parameter("mission_stop_override", True)

        # DRIVER_ONLY_ROUTE_AGENT:
        # Route agent trafik ışığı/tabela/yasak dönüş kararı vermez.
        # Sadece final decision mesajındaki decision + target_speed alanını uygular.
        self.declare_parameter("driver_only_decision_mode", True)


        self.carla_root = self.get_parameter("carla_root").value
        self.host = self.get_parameter("host").value
        self.port = int(self.get_parameter("port").value)
        self.timeout = float(self.get_parameter("timeout").value)
        self.ego_role_name = self.get_parameter("ego_role_name").value

        self.decision_topic = self.get_parameter("decision_topic").value
        self.mission_topic = self.get_parameter("mission_topic").value

        self.use_planner_local_target = bool(
            self.get_parameter("use_planner_local_target").value
        )
        self.local_target_topic = self.get_parameter("local_target_topic").value
        self.planner_fresh_timeout_s = float(
            self.get_parameter("planner_fresh_timeout_s").value
        )
        self.planner_speed_hint_enabled = bool(
            self.get_parameter("planner_speed_hint_enabled").value
        )
        self.planner_local_target_as_destination = bool(
            self.get_parameter("planner_local_target_as_destination").value
        )
        self.planner_destination_hold_enabled = bool(
            self.get_parameter("planner_destination_hold_enabled").value
        )
        self.planner_destination_reached_m = float(
            self.get_parameter("planner_destination_reached_m").value
        )
        self.planner_destination_update_min_distance_m = float(
            self.get_parameter("planner_destination_update_min_distance_m").value
        )
        self.planner_destination_update_min_interval_s = float(
            self.get_parameter("planner_destination_update_min_interval_s").value
        )
        self.last_agent_destination_set_time = 0.0
        self.active_planner_route_id = None
        self.planner_target_key_resolution_m = float(
            self.get_parameter("planner_target_key_resolution_m").value
        )
        self.target_speed_smoothing_enabled = bool(
            self.get_parameter("target_speed_smoothing_enabled").value
        )
        self.target_speed_accel_limit_mps2 = float(
            self.get_parameter("target_speed_accel_limit_mps2").value
        )
        self.target_speed_decel_limit_mps2 = float(
            self.get_parameter("target_speed_decel_limit_mps2").value
        )
        self.green_release_min_start_speed_mps = float(
            self.get_parameter("green_release_min_start_speed_mps").value
        )
        self.tl_stopline_stop_before_m = float(
            self.get_parameter("tl_stopline_stop_before_m").value
        )
        self.tl_passed_stopline_ignore_s = float(
            self.get_parameter("tl_passed_stopline_ignore_s").value
        )
        self.tl_front_bumper_offset_m = float(
            self.get_parameter("tl_front_bumper_offset_m").value
        )
        self.tl_desired_front_bumper_stop_before_m = float(
            self.get_parameter("tl_desired_front_bumper_stop_before_m").value
        )

        self.manual_tl_stoplines_enabled = bool(
            self.get_parameter("manual_tl_stoplines_enabled").value
        )
        self.manual_tl_stoplines_path = str(
            self.get_parameter("manual_tl_stoplines_path").value
        )

        self.tl_stop_hold_active = False
        self.tl_stop_hold_until = 0.0
        self.active_tl_stopline_id = None
        self.active_tl_stopline_line = None
        self.active_tl_stopline_line_dist = None
        self.active_tl_stopline_control_dist = None
        self.active_tl_stopline_lat = None
        self.red_hold_active = False
        self.red_hold_until_green = False
        self.no_stopline_red_hold_since = 0.0
        self.no_stopline_last_red_yellow_time = 0.0
        self.no_stopline_decision_go_since = 0.0
        self.last_tl_stopline_debug = {}
        self.green_release_force_until = 0.0
        self.green_release_started_at = 0.0
        self.tl_green_release_active = False
        self.tl_green_release_from_hold = False
        self.tl_hold_safe_go_release_s = 0.6
        self.tl_hold_go_candidate_since = 0.0
        self.tl_post_green_ignore_until = 0.0
        self.tl_passed_stopline_ignore_until = 0.0
        self.tl_recently_passed_stoplines = {}
        self.tl_released_stoplines = {}
        self.tl_verified_green_time = 0.0
        self.tl_last_reliable_state = "unknown"
        self.tl_last_reliable_state_time = 0.0
        self.tl_last_reliable_stopline_id = None
        self.tl_active_green_count = 0
        self.tl_active_green_since = 0.0
        self.tl_active_green_last_time = 0.0
        self.tl_active_green_stopline_id = None
        self.tl_last_green_seen_time = 0.0
        self.tl_green_seen_count = 0
        self.tl_overshoot_recover_until = 0.0
        self.tl_waiting_for_verified_green = False
        self.tl_wait_stopline_id = None
        self.tl_wait_started_after_red_stop = False
        self.tl_recent_green_blocks_red_until = 0.0

        self.last_smoothed_target_speed_mps = None
        self.last_smoothed_target_time = 0.0

        self.debug_topic = self.get_parameter("debug_topic").value
        self.collision_topic = self.get_parameter("collision_topic").value

        self.collision_halt_s = float(self.get_parameter("collision_halt_s").value)
        self.collision_until = 0.0
        self.last_collision = None

        self.max_speed_mps = float(self.get_parameter("max_speed_mps").value)
        self.go_speed_mps = float(self.get_parameter("go_speed_mps").value)
        self.slow_speed_mps = float(self.get_parameter("slow_speed_mps").value)
        self.parking_speed_mps = float(self.get_parameter("parking_speed_mps").value)
        self.max_steer = float(self.get_parameter("max_steer").value)

        self.lane_assist_enabled = bool(self.get_parameter("lane_assist_enabled").value)
        self.lane_topic = self.get_parameter("lane_topic").value
        self.lane_min_confidence = float(self.get_parameter("lane_min_confidence").value)
        self.lane_fresh_timeout_s = float(self.get_parameter("lane_fresh_timeout_s").value)
        self.lane_blend_straight = float(self.get_parameter("lane_blend_straight").value)
        self.lane_blend_turn = float(self.get_parameter("lane_blend_turn").value)
        self.lane_turn_steer_threshold = float(self.get_parameter("lane_turn_steer_threshold").value)
        self.lane_allowed_stages = [
            x.strip() for x in str(self.get_parameter("lane_allowed_stages").value).split(",") if x.strip()
        ]

        self.mission_stop_override = bool(self.get_parameter("mission_stop_override").value)
        self.driver_only_decision_mode = bool(self.get_parameter("driver_only_decision_mode").value)




        # km/h parametreleri verilirse iç m/s değerlerini burada override et.
        self.max_speed_kmh = float(self.get_parameter("max_speed_kmh").value)
        self.go_speed_kmh = float(self.get_parameter("go_speed_kmh").value)
        self.slow_speed_kmh = float(self.get_parameter("slow_speed_kmh").value)
        self.parking_speed_kmh = float(self.get_parameter("parking_speed_kmh").value)

        if self.max_speed_kmh >= 0.0:
            self.max_speed_mps = self.kmh_to_mps(self.max_speed_kmh)
        if self.go_speed_kmh >= 0.0:
            self.go_speed_mps = self.kmh_to_mps(self.go_speed_kmh)
        if self.slow_speed_kmh >= 0.0:
            self.slow_speed_mps = self.kmh_to_mps(self.slow_speed_kmh)
        if self.parking_speed_kmh >= 0.0:
            self.parking_speed_mps = self.kmh_to_mps(self.parking_speed_kmh)


        self.carla = load_carla(self.carla_root)
        self.client = self.carla.Client(self.host, self.port)
        self.client.set_timeout(self.timeout)
        self.world = self.client.get_world()
        self.map = self.world.get_map()
        self.manual_tl_stoplines = self.load_manual_tl_stoplines()
        self.ego = self.wait_for_ego()

        self.BasicAgent = self.load_basic_agent()
        self.agent = self.BasicAgent(self.ego, target_speed=self.go_speed_mps * 3.6)

        self.configure_agent_ignore_rules()

        self.latest_decision = {
            "decision": "STOP",
            "risk": "UNKNOWN",
            "target_speed": 0.0,
            "reason": "initial",
        }

        self.latest_mission = None
        self.last_decision_time = 0.0
        self.last_mission_time = 0.0

        self.latest_local_target = None
        self.last_local_target_time = 0.0

        self.active_target_key = None
        self.active_destination = None
        self.route_status = "not_planned"

        self.latest_lane = None
        self.last_lane_time = 0.0
        self.current_lane_debug = {
            "enabled": self.lane_assist_enabled,
            "used": False,
            "reason": "initial",
        }

        self.debug_pub = self.create_publisher(String, self.debug_topic, 10)

        self.create_subscription(String, self.decision_topic, self.decision_cb, 10)
        self.create_subscription(String, self.mission_topic, self.mission_cb, 10)
        self.create_subscription(String, self.local_target_topic, self.local_target_cb, 10)
        self.create_subscription(String, self.collision_topic, self.collision_cb, 10)
        self.create_subscription(String, self.lane_topic, self.lane_cb, 10)

        rate = float(self.get_parameter("control_rate_hz").value)
        self.timer = self.create_timer(1.0 / max(rate, 1.0), self.tick)

        self.get_logger().info("TEKNOFEST route agent hazır: CARLA BasicAgent lane follower aktif.")

    def load_basic_agent(self):
        possible_paths = [
            os.path.join(self.carla_root, "PythonAPI", "carla"),
            os.path.join(self.carla_root, "PythonAPI"),
            os.path.expanduser("~/CARLA_DISK/PythonAPI/carla"),
            os.path.expanduser("~/İndirilenler/PythonAPI/carla"),
        ]

        for p in possible_paths:
            if os.path.isdir(p) and p not in sys.path:
                sys.path.append(p)

        try:
            from agents.navigation.basic_agent import BasicAgent
            self.get_logger().info("BasicAgent import OK.")
            return BasicAgent
        except Exception as e:
            raise RuntimeError(f"BasicAgent import edilemedi. PythonAPI/carla/agents yolu yok veya hatalı: {e}")

    def configure_agent_ignore_rules(self):
        for method_name in ["ignore_traffic_lights", "ignore_stop_signs", "ignore_vehicles"]:
            try:
                if hasattr(self.agent, method_name):
                    getattr(self.agent, method_name)(True)
                    self.get_logger().info(f"BasicAgent {method_name}(True)")
            except Exception as e:
                self.get_logger().warning(f"{method_name} ayarlanamadı: {e}")

    def wait_for_ego(self):
        for _ in range(300):
            vehicles = self.world.get_actors().filter("vehicle.*")
            for vehicle in vehicles:
                if vehicle.attributes.get("role_name", "") == self.ego_role_name:
                    self.get_logger().info(f"Ego bulundu: id={vehicle.id}")
                    return vehicle
            time.sleep(0.2)

        raise RuntimeError("Ego vehicle bulunamadı.")

    def decision_cb(self, msg):
        try:
            data = json.loads(msg.data)
            now = time.time()
            new_reason = str(data.get("reason", "") or "").lower()
            new_state = str(data.get("traffic_light_state", "") or "").lower()
            new_decision = str(data.get("decision", "") or "").upper()
            green_evidence_now = (
                new_state == "green"
                or "green_light_weak_not_latched_release" in new_reason
                or "green_light_controlled_release" in new_reason
                or "green_light_confirmed_stable" in new_reason
            )

            go_candidate = (
                new_decision == "GO"
                and new_state not in {"red", "yellow"}
                and "red_light" not in new_reason
                and "yellow_light" not in new_reason
            )
            red_yellow_now = (
                new_state in {"red", "yellow"}
                or "red_light" in new_reason
                or "yellow_light" in new_reason
            )

            if go_candidate:
                if float(getattr(self, "tl_hold_go_candidate_since", 0.0) or 0.0) <= 0.0:
                    self.tl_hold_go_candidate_since = now
                if float(getattr(self, "no_stopline_decision_go_since", 0.0) or 0.0) <= 0.0:
                    self.no_stopline_decision_go_since = now
            else:
                self.tl_hold_go_candidate_since = 0.0
                self.no_stopline_decision_go_since = 0.0

            if green_evidence_now:
                self.tl_last_green_seen_time = now
                self.tl_green_seen_count = int(getattr(self, "tl_green_seen_count", 0) or 0) + 1
                self.tl_recent_green_blocks_red_until = now + 0.8
            elif red_yellow_now:
                if now > float(getattr(self, "tl_recent_green_blocks_red_until", 0.0) or 0.0):
                    self.tl_green_seen_count = 0

            if (
                red_yellow_now
                and not green_evidence_now
                and now > float(getattr(self, "tl_recent_green_blocks_red_until", 0.0) or 0.0)
            ):
                self.no_stopline_last_red_yellow_time = now
                self.tl_last_reliable_state = new_state if new_state in {"red", "yellow"} else "red"
                self.tl_last_reliable_state_time = now
                self.reset_active_green_confirmation()

            if self.is_verified_green_decision(data, new_reason):
                self.tl_verified_green_time = now
                self.tl_last_reliable_state = "green"
                self.tl_last_reliable_state_time = now
                self.release_wait_green_latch()
                if not bool(getattr(self, "active_tl_stopline_id", None)):
                    self.clear_red_hold(mark_green_release=True)

                self.tl_post_green_ignore_until = now + 4.0
                self.last_brake_cmd = 0.0
                self.last_throttle_cmd = 0.0

            red_stop_now = (
                "red_light" in new_reason
                and (
                    "decision_stop" in new_reason
                    or "overhead_stop" in new_reason
                    or "tl_stopline_stop" in new_reason
                )
            )

            if red_stop_now:
                self.tl_green_release_active = False
                self.tl_green_release_from_hold = False
                self.green_release_force_until = 0.0
                self.green_release_started_at = 0.0
                self.red_hold_active = True
                self.red_hold_until_green = True
                self.tl_stop_hold_active = True
                # Kırmızı döngü uzun sürerse bile güvenli tarafta kal.
                # Yeşil confirmed geldiğinde zaten hemen temizlenir.
                self.tl_stop_hold_until = now + 45.0

            self.latest_decision = data
            self.last_decision_time = now
        except Exception as exc:
            self.get_logger().warning(f"decision parse hatası: {exc}")

    def angle_norm_deg(self, angle):
        return (float(angle) + 180.0) % 360.0 - 180.0

    def classify_relative_turn_to_location(self, ego_tf, loc):
        try:
            ego_loc = ego_tf.location
            fwd = ego_tf.get_forward_vector()
            right = ego_tf.get_right_vector()

            vx = float(loc.x) - float(ego_loc.x)
            vy = float(loc.y) - float(ego_loc.y)

            forward_dot = vx * float(fwd.x) + vy * float(fwd.y)
            right_dot = vx * float(right.x) + vy * float(right.y)

            angle_deg = math.degrees(math.atan2(right_dot, max(0.001, forward_dot)))

            if angle_deg > 25.0:
                return "right", angle_deg
            if angle_deg < -25.0:
                return "left", angle_deg
            return "straight", angle_deg

        except Exception:
            return "unknown", 0.0

    def mission_cb(self, msg):
        try:
            self.latest_mission = json.loads(msg.data)
            self.last_mission_time = time.time()
        except Exception as exc:
            self.get_logger().warning(f"mission parse hatası: {exc}")

    def local_target_cb(self, msg):
        """
        GlobalRoutePlannerNode çıktısını RouteAgent hedef formatına çevirir.

        Gelen mesaj:
          /adas/planning/local_target

        Bu hedef, GeoJSON'dan gelen görev noktası değildir.
        Runtime planner'ın ürettiği yakın takip hedefidir.
        """
        try:
            data = json.loads(msg.data)

            x = data.get("x")
            y = data.get("y")
            z = data.get("z", 0.2)

            if x is None or y is None:
                return

            route_id = data.get("route_id")
            target_name = str(data.get("target_name", "planner_target"))
            road_id = data.get("road_id")
            lane_id = data.get("lane_id")

            local = dict(data)
            local["name"] = f"planner_local_{target_name}"
            local["description"] = "Runtime planner local target"
            local["carla_x"] = float(x)
            local["carla_y"] = float(y)
            local["carla_z"] = float(z)
            local["carla_yaw"] = float(data.get("yaw", 0.0))
            local["lat"] = float(y)
            local["lon"] = float(x)
            local["kind"] = "planner_local"
            key_res = max(0.5, float(self.planner_target_key_resolution_m))
            x_bucket = int(round(float(x) / key_res))
            y_bucket = int(round(float(y) / key_res))

            local["_planner_key"] = (
                f"planner|route={route_id}|target={target_name}|"
                f"xb={x_bucket}|yb={y_bucket}|res={key_res:.1f}|"
                f"road={road_id}|lane={lane_id}"
            )

            self.latest_local_target = local
            self.last_local_target_time = time.time()

        except Exception as exc:
            self.get_logger().warning(f"planner local_target parse hatası: {exc}")

    def collision_cb(self, msg):
        self.last_collision = msg.data
        self.collision_until = time.time() + self.collision_halt_s
        self.get_logger().warning(f"COLLISION HALT: {msg.data}")

    def lane_cb(self, msg):
        try:
            self.latest_lane = json.loads(msg.data)
            self.last_lane_time = time.time()
        except Exception as exc:
            self.get_logger().warning(f"lane assist parse hatası: {exc}")

    def apply_lane_assist_to_steer(self, basic_steer, target_speed):
        now = time.time()
        stage = self.get_stage()

        debug = {
            "enabled": bool(self.lane_assist_enabled),
            "used": False,
            "reason": "not_used",
            "basic_steer": round(float(basic_steer), 4),
            "final_steer": round(float(basic_steer), 4),
            "lane_confidence": None,
            "lane_steer": None,
            "lane_offset_norm": None,
            "blend": 0.0,
        }

        if not self.lane_assist_enabled:
            debug["reason"] = "disabled"
            self.current_lane_debug = debug
            return basic_steer

        if stage not in self.lane_allowed_stages:
            debug["reason"] = f"stage_not_allowed:{stage}"
            self.current_lane_debug = debug
            return basic_steer

        if target_speed <= 0.05:
            debug["reason"] = "target_speed_zero"
            self.current_lane_debug = debug
            return basic_steer

        if self.latest_lane is None or now - self.last_lane_time > self.lane_fresh_timeout_s:
            debug["reason"] = "lane_timeout"
            self.current_lane_debug = debug
            return basic_steer

        lane_detected = bool(self.latest_lane.get("lane_detected", False))
        conf = float(self.latest_lane.get("confidence", 0.0))
        lane_steer = float(self.latest_lane.get("lane_steer", 0.0))
        offset_norm = float(self.latest_lane.get("offset_norm", 0.0))

        debug["lane_confidence"] = round(conf, 3)
        debug["lane_steer"] = round(lane_steer, 4)
        debug["lane_offset_norm"] = round(offset_norm, 4)

        if not lane_detected:
            debug["reason"] = "lane_not_detected"
            self.current_lane_debug = debug
            return basic_steer

        if conf < self.lane_min_confidence:
            debug["reason"] = f"low_conf:{conf:.3f}"
            self.current_lane_debug = debug
            return basic_steer

        # Keskin dönüşte lane etkisini azalt. Düz yolda daha fazla hizalasın.
        if abs(basic_steer) >= self.lane_turn_steer_threshold:
            blend = self.lane_blend_turn
            debug["reason"] = "used_turn_low_blend"
        else:
            blend = self.lane_blend_straight
            debug["reason"] = "used_straight_blend"

        blend = self.clamp(blend, 0.0, 0.75)
        final_steer = (1.0 - blend) * basic_steer + blend * lane_steer
        final_steer = self.clamp(final_steer, -self.max_steer, self.max_steer)

        debug["used"] = True
        debug["blend"] = round(float(blend), 3)
        debug["final_steer"] = round(float(final_steer), 4)

        self.current_lane_debug = debug
        return final_steer

    def get_speed(self):
        v = self.ego.get_velocity()
        return math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)

    def clamp(self, value, mn, mx):
        return max(mn, min(mx, float(value)))

    def mps_to_kmh(self, value):
        try:
            return float(value) * 3.6
        except Exception:
            return 0.0

    def kmh_to_mps(self, value):
        try:
            return float(value) / 3.6
        except Exception:
            return 0.0


    def hard_stop_control(self):
        control = self.carla.VehicleControl()
        control.throttle = 0.0
        control.brake = 1.0
        control.steer = 0.0
        control.hand_brake = False
        control.reverse = False
        return control

    def load_manual_tl_stoplines(self):
        if not getattr(self, "manual_tl_stoplines_enabled", False):
            return []

        path = str(getattr(self, "manual_tl_stoplines_path", "") or "").strip()
        if not path:
            return []

        path = os.path.expanduser(path)
        if not os.path.isabs(path):
            path = os.path.join(os.getcwd(), path)

        if not os.path.exists(path):
            self.get_logger().warning(
                f"manual_tl_stoplines_path bulunamadı, stopline devre dışı: {path}"
            )
            return []

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            stoplines = data.get("stoplines", data if isinstance(data, list) else [])
            normalized = []

            for item in stoplines:
                try:
                    def _maybe_int(v):
                        try:
                            return int(v) if v is not None else None
                        except Exception:
                            return None

                    normalized.append({
                        "id": str(item.get("id", f"stopline_{len(normalized)+1}")),
                        "map": str(item.get("map", "")),
                        "x": float(item["x"]),
                        "y": float(item["y"]),
                        "yaw_deg": float(item["yaw_deg"]),

                        # CARLA eşleştirme için bu alanları koru.
                        "road_id": _maybe_int(item.get("road_id")),
                        "section_id": _maybe_int(item.get("section_id")),
                        "lane_id": _maybe_int(item.get("lane_id")),
                        "s": float(item.get("s", 0.0) or 0.0),

                        "approach_m": float(item.get("approach_m", 80.0)),
                        "crawl_m": float(item.get("crawl_m", 10.0)),
                        "stop_margin_m": float(item.get("stop_margin_m", 1.50)),
                        "release_after_pass_m": float(item.get("release_after_pass_m", 4.0)),
                        "lateral_half_width_m": float(item.get("lateral_half_width_m", 4.2)),
                        "stop_before_m": float(item.get("stop_before_m", self.tl_stopline_stop_before_m)),
                        "stop_speed_mps": float(item.get("stop_speed_mps", 0.0)),
                        "crawl_speed_mps": float(item.get("crawl_speed_mps", 1.10)),
                        "approach_speed_mps": float(item.get("approach_speed_mps", 4.00)),
                    })
                except Exception as exc:
                    self.get_logger().warning(f"manual stopline parse skip: {exc} item={item}")

            self.get_logger().info(f"Manual TL stoplines loaded: {len(normalized)} from {path}")
            return normalized

        except Exception as exc:
            self.get_logger().warning(f"Manual TL stoplines yüklenemedi: {exc}")
            return []

    def get_ego_front_xy(self):
        tf = self.ego.get_transform()
        loc = tf.location
        yaw = math.radians(float(tf.rotation.yaw))
        extent_x = float(getattr(self, "tl_front_bumper_offset_m", 2.3) or 2.3)

        if extent_x <= 0.0:
            try:
                extent_x = float(self.ego.bounding_box.extent.x)
            except Exception:
                extent_x = 2.25

        front_x = float(loc.x) + math.cos(yaw) * extent_x
        front_y = float(loc.y) + math.sin(yaw) * extent_x
        return front_x, front_y

    def manual_tl_red_active(self):
        d = getattr(self, "latest_decision", {}) or {}

        decision = str(d.get("decision", "") or "").upper()
        state = str(d.get("traffic_light_state", "") or "").lower()
        reason = str(d.get("reason", "") or "").lower()
        latched = bool(d.get("red_light_latch_active", False))

        if bool(getattr(self, "tl_waiting_for_verified_green", False)):
            return True

        current_red_yellow = (
            state in {"red", "yellow"}
            or "red_light" in reason
            or "yellow_light" in reason
            or "traffic_light_red" in reason
            or "traffic_light_yellow" in reason
        )

        if decision == "GO" and not current_red_yellow:
            return False

        if "green_light_confirmed_stable" in reason:
            return False

        if state == "green" and "red_light" not in reason and not latched:
            return False

        return (
            state == "red"
            or latched
            or "red_light" in reason
        )

    def mark_stopline_passed(self, line_id):
        now = time.time()
        line_id = str(line_id)
        ignore_until = now + max(5.0, float(getattr(self, "tl_passed_stopline_ignore_s", 4.0)))
        self.tl_passed_stopline_ignore_until = max(
            float(getattr(self, "tl_passed_stopline_ignore_until", 0.0) or 0.0),
            ignore_until,
        )
        recent = getattr(self, "tl_recently_passed_stoplines", {}) or {}
        recent[line_id] = ignore_until
        canonical_id = self.canonical_tl_stopline_id(line_id)
        if canonical_id:
            recent[canonical_id] = ignore_until
        self.tl_recently_passed_stoplines = {
            str(k): float(v)
            for k, v in recent.items()
            if float(v) > now
        }
        if str(getattr(self, "active_tl_stopline_id", "") or "") == line_id:
            self.clear_red_hold()

    def canonical_tl_stopline_id(self, line_id):
        line_id = str(line_id or "")
        for prefix in (
            "virtual_stopline_passed_junction_",
            "virtual_stopline_junction_entry_",
        ):
            if line_id.startswith(prefix):
                return "virtual_stopline_junction_" + line_id[len(prefix):]
        return line_id

    def mark_stopline_released(self, line_id):
        now = time.time()
        line_id = str(line_id or "")
        if not line_id:
            return
        until = now + max(5.0, float(getattr(self, "tl_passed_stopline_ignore_s", 4.0)))
        released = getattr(self, "tl_released_stoplines", {}) or {}
        released[line_id] = until
        canonical_id = self.canonical_tl_stopline_id(line_id)
        if canonical_id:
            released[canonical_id] = until
        self.tl_released_stoplines = {
            str(k): float(v)
            for k, v in released.items()
            if float(v) > now
        }

    def stopline_recently_passed(self, line_id=None):
        now = time.time()
        if line_id:
            key = str(line_id)
            canonical_id = self.canonical_tl_stopline_id(key)
            recent = getattr(self, "tl_recently_passed_stoplines", {}) or {}
            return (
                now < float(recent.get(key, 0.0) or 0.0)
                or now < float(recent.get(canonical_id, 0.0) or 0.0)
            )
        return now < float(getattr(self, "tl_passed_stopline_ignore_until", 0.0) or 0.0)

    def stopline_recently_released(self, line_id):
        now = time.time()
        key = str(line_id or "")
        canonical_id = self.canonical_tl_stopline_id(key)
        released = getattr(self, "tl_released_stoplines", {}) or {}
        return (
            now < float(released.get(key, 0.0) or 0.0)
            or now < float(released.get(canonical_id, 0.0) or 0.0)
        )

    def selected_stopline_passed_ignore(self):
        line_id = str(getattr(self, "active_tl_stopline_id", "") or "")
        if not line_id:
            return False
        recent = getattr(self, "tl_recently_passed_stoplines", {}) or {}
        canonical_id = self.canonical_tl_stopline_id(line_id)
        return (
            time.time() < float(recent.get(line_id, 0.0) or 0.0)
            or time.time() < float(recent.get(canonical_id, 0.0) or 0.0)
        )

    def tl_wait_green_release_allowed(self, reason=None):
        latest = getattr(self, "latest_decision", {}) or {}
        state = str(latest.get("traffic_light_state", "") or "").lower()
        text = (str(reason or "") + "|" + str(latest.get("reason", "") or "")).lower()
        green_count = int(getattr(self, "tl_green_seen_count", 0) or 0)
        recent_green = (
            time.time() - float(getattr(self, "tl_last_green_seen_time", 0.0) or 0.0) <= 1.0
            and green_count >= 1
        )
        weak_green_debounced = (
            "green_light_weak_not_latched_release" in text
            and recent_green
            and green_count >= 2
        )
        return (
            state == "green"
            or "green_light_confirmed_stable" in text
            or "green_light_controlled_release" in text
            or weak_green_debounced
            or (
                recent_green
                and green_count >= 2
                and str(getattr(self, "tl_active_green_stopline_id", "") or "") in {
                    "",
                    str(getattr(self, "tl_wait_stopline_id", "") or ""),
                    str(getattr(self, "active_tl_stopline_id", "") or ""),
                }
            )
        )

    def tl_wait_green_release_reason_tag(self, reason=None):
        text = (str(reason or "") + "|" + str((getattr(self, "latest_decision", {}) or {}).get("reason", "") or "")).lower()
        if (
            "green_light_weak_not_latched_release" in text
            and int(getattr(self, "tl_green_seen_count", 0) or 0) >= 2
        ):
            return "tl_wait_green_latch_release_weak_green_debounced"
        return "tl_wait_green_latch_release_green_evidence"

    def set_wait_green_latch(self, line_id=None, metrics=None, reason=None):
        line_id = str(line_id or getattr(self, "active_tl_stopline_id", "") or "")
        if not line_id:
            return
        self.tl_waiting_for_verified_green = True
        self.tl_wait_stopline_id = line_id
        self.tl_wait_started_after_red_stop = True
        self.red_hold_active = True
        self.red_hold_until_green = True
        self.tl_stop_hold_active = True
        self.tl_stop_hold_until = time.time() + 45.0
        self.tl_overshoot_recover_until = max(
            float(getattr(self, "tl_overshoot_recover_until", 0.0) or 0.0),
            time.time() + 1.0,
        )
        if isinstance(metrics, dict):
            self.update_stopline_debug(line_id, metrics, reason or "")
            self.last_tl_stopline_debug["passed_ignore"] = False

    def wait_green_latch_reason(self, reason=None, metrics=None):
        state = self.latest_tl_state()
        tag = "tl_wait_green_latch_hold"
        if state == "unknown":
            tag = "tl_wait_green_latch_keep_on_unknown"
        control_dist = None
        line_dist = None
        lat = None
        if isinstance(metrics, dict):
            control_dist = metrics.get("control_dist")
            line_dist = metrics.get("line_dist")
            lat = metrics.get("lat")
        base = str(reason or "")
        for dup in (
            "tl_wait_green_latch_hold",
            "tl_wait_green_latch_keep_on_unknown",
            "tl_stationary_hold_no_rollback",
            "tl_crossed_release_blocked_wait_green",
        ):
            base = base.replace("|" + dup, "")
        parts = [
            base,
            tag,
            "tl_stationary_hold_no_rollback",
        ]
        try:
            if control_dist is not None and float(control_dist) < 0.0:
                parts.append("tl_crossed_release_blocked_wait_green")
        except Exception:
            pass
        detail = f"id={getattr(self, 'tl_wait_stopline_id', None)}"
        try:
            if line_dist is not None:
                detail += f",line_dist={float(line_dist):.2f}"
            if control_dist is not None:
                detail += f",control_dist={float(control_dist):.2f}"
            if lat is not None:
                detail += f",lat={float(lat):.2f}"
        except Exception:
            pass
        return "|".join([p for p in parts if p]) + ":" + detail

    def release_wait_green_latch(self):
        self.tl_waiting_for_verified_green = False
        self.tl_wait_stopline_id = None
        self.tl_wait_started_after_red_stop = False

    def current_tl_red_yellow(self, reason=None):
        latest = getattr(self, "latest_decision", {}) or {}
        state = str(latest.get("traffic_light_state", "") or "").lower()
        text = (str(reason or "") + "|" + str(latest.get("reason", "") or "")).lower()
        return (
            state in {"red", "yellow"}
            or "red_light" in text
            or "yellow_light" in text
            or "traffic_light_red" in text
            or "traffic_light_yellow" in text
        )

    def traffic_light_decision_text(self, text):
        text = str(text or "").lower()
        return (
            "red_light_no_sensor_crawl_visual" in text
            or "red_light_no_sensor_overhead_stop" in text
            or "red_light_detected" in text
            or "yellow_light_detected" in text
            or "traffic_light" in text
            or "tl_red" in text
            or "tl_yellow" in text
            or "red_light_" in text
            or "yellow_light" in text
        )

    def non_tl_safety_text(self, text):
        text = str(text or "").lower()
        if "front_vehicle" in text and "front_vehicle_not_found" not in text:
            return True
        return (
            "pedestrian" in text
            or "person" in text
            or "obstacle" in text
            or "collision" in text
            or "mission_stop" in text
            or "passenger" in text
            or "route_missing" in text
            or "timeout" in text
        )

    def passed_stopline_override_active(self, reason=None, use_debug_passed=True):
        if not use_debug_passed:
            return False

        debug = getattr(self, "last_tl_stopline_debug", {}) or {}
        selected_id = str(debug.get("selected_stopline_id", "") or "")

        try:
            line_dist = debug.get("line_dist")
            line_dist = None if line_dist is None else float(line_dist)
        except Exception:
            line_dist = None

        try:
            control_dist = debug.get("control_dist")
            control_dist = None if control_dist is None else float(control_dist)
        except Exception:
            control_dist = None

        real_passed = (
            selected_id.startswith("virtual_stopline_passed")
            or (line_dist is not None and line_dist <= -3.0)
            or (control_dist is not None and control_dist <= -5.0)
        )
        if self.current_tl_red_yellow(reason) and time.time() < float(
            getattr(self, "tl_overshoot_recover_until", 0.0) or 0.0
        ) and not real_passed:
            return False
        if self.current_tl_red_yellow(reason) and not real_passed:
            return False

        return bool(debug.get("passed_ignore", False)) or real_passed

    def maybe_override_passed_tl_decision(self, target_speed, reason, use_debug_passed=True):
        if bool(getattr(self, "tl_waiting_for_verified_green", False)) and not self.tl_wait_green_release_allowed(reason):
            return None
        text = str(reason or "").lower()
        if not self.traffic_light_decision_text(text):
            return None
        if self.non_tl_safety_text(text):
            return None
        if not self.passed_stopline_override_active(reason, use_debug_passed=use_debug_passed):
            return None

        self.red_hold_active = False
        self.red_hold_until_green = False
        self.tl_stop_hold_active = False
        self.tl_stop_hold_until = 0.0

        tag = "tl_passed_ignore_override_visual_red"
        if "yellow" in text or "traffic_light_yellow" in text:
            tag = "tl_passed_ignore_override_decision_tl_stop"

        speed = self.clamp(float(getattr(self, "go_speed_mps", 0.0) or 0.0), 0.0, self.max_speed_mps)
        if speed <= 0.05:
            speed = self.clamp(float(target_speed or 0.0), 0.0, self.max_speed_mps)
        if speed <= 0.05:
            speed = self.clamp(8.33, 0.0, self.max_speed_mps)

        return speed, f"driver_only_decision_go_after_passed_tl|{tag}"

    def clear_red_hold(self, mark_green_release=False):
        had_hold = (
            bool(getattr(self, "red_hold_active", False))
            or bool(getattr(self, "tl_stop_hold_active", False))
            or bool(getattr(self, "active_tl_stopline_id", None))
            or float(getattr(self, "no_stopline_red_hold_since", 0.0) or 0.0) > 0.0
        )
        self.red_hold_active = False
        self.red_hold_until_green = False
        self.tl_stop_hold_active = False
        self.tl_stop_hold_until = 0.0
        active_id = self.active_tl_stopline_id
        if mark_green_release and active_id:
            self.mark_stopline_released(active_id)
        self.active_tl_stopline_id = None
        self.active_tl_stopline_line = None
        self.active_tl_stopline_line_dist = None
        self.active_tl_stopline_control_dist = None
        self.active_tl_stopline_lat = None
        self.no_stopline_red_hold_since = 0.0
        self.no_stopline_last_red_yellow_time = 0.0
        self.no_stopline_decision_go_since = 0.0
        self.reset_active_green_confirmation()
        self.tl_overshoot_recover_until = 0.0
        self.tl_last_reliable_stopline_id = None
        if mark_green_release:
            self.release_wait_green_latch()
        self.last_tl_stopline_debug = {
            "selected_stopline_id": None,
            "line_dist": None,
            "control_dist": None,
            "stop_before": None,
            "lat": None,
            "red_hold_active": False,
            "passed_ignore": False,
            "tl_state": self.latest_tl_state(),
            "reason": "",
        }
        self.last_brake_cmd = 0.0
        if mark_green_release and had_hold:
            now = time.time()
            self.green_release_started_at = now
            self.green_release_force_until = now + 2.0
            self.tl_green_release_active = True
            self.tl_green_release_from_hold = True
        elif not mark_green_release:
            self.green_release_started_at = 0.0
            self.green_release_force_until = 0.0
            self.tl_green_release_active = False
            self.tl_green_release_from_hold = False

    def set_active_tl_stopline(self, line, line_dist, lat):
        line_id = str(line.get("id", "manual_stopline"))
        if str(getattr(self, "active_tl_stopline_id", "") or "") != line_id:
            self.reset_active_green_confirmation()
        stop_before = float(line.get("stop_before_m", self.tl_stopline_stop_before_m))
        control_dist = float(line_dist) - stop_before
        self.active_tl_stopline_id = line_id
        self.active_tl_stopline_line = dict(line)
        self.active_tl_stopline_line_dist = float(line_dist)
        self.active_tl_stopline_control_dist = float(control_dist)
        self.active_tl_stopline_lat = float(lat)
        reliable_red_yellow = self.current_tl_red_yellow()
        memory_red_yellow = (
            str(getattr(self, "active_tl_stopline_id", "") or "") == line_id
            and str(getattr(self, "tl_last_reliable_state", "") or "").lower() in {"red", "yellow"}
            and float(control_dist) > -5.0
        )
        hold_now = bool(control_dist <= 30.0 or reliable_red_yellow or memory_red_yellow)
        self.red_hold_active = hold_now
        self.red_hold_until_green = hold_now
        self.tl_stop_hold_active = hold_now
        self.tl_stop_hold_until = time.time() + 45.0 if hold_now else 0.0
        if reliable_red_yellow:
            state = self.latest_tl_state()
            self.tl_last_reliable_state = state if state in {"red", "yellow"} else "red"
            self.tl_last_reliable_state_time = time.time()
            self.tl_last_reliable_stopline_id = line_id
        self.last_tl_stopline_debug = {
            "selected_stopline_id": line_id,
            "line_dist": round(float(line_dist), 3),
            "control_dist": round(float(control_dist), 3),
            "stop_before": round(float(stop_before), 3),
            "lat": round(float(lat), 3),
            "red_hold_active": hold_now,
            "passed_ignore": self.selected_stopline_passed_ignore(),
            "tl_state": self.latest_tl_state(),
        }

    def latest_tl_state(self):
        latest = getattr(self, "latest_decision", {}) or {}
        return str(latest.get("traffic_light_state", "unknown") or "unknown").lower()

    def tl_state_confidence(self, data=None):
        data = data if isinstance(data, dict) else (getattr(self, "latest_decision", {}) or {})
        for key in ("traffic_light_state_confidence", "tl_state_confidence", "state_confidence", "state_conf"):
            try:
                value = data.get(key)
                if value is not None:
                    return float(value)
            except Exception:
                pass
        tl = data.get("traffic_light") if isinstance(data, dict) else None
        if isinstance(tl, dict):
            for key in ("traffic_light_state_confidence", "tl_state_confidence", "state_confidence", "state_conf"):
                try:
                    value = tl.get(key)
                    if value is not None:
                        return float(value)
                except Exception:
                    pass
        return None

    def is_verified_green_decision(self, data=None, reason=None):
        data = data if isinstance(data, dict) else (getattr(self, "latest_decision", {}) or {})
        text = (str(reason or "") + "|" + str(data.get("reason", "") or "")).lower()
        state = str(data.get("traffic_light_state", "unknown") or "unknown").lower()
        if state != "green" or "green_light_confirmed_stable" not in text:
            return False
        conf = self.tl_state_confidence(data)
        return conf is None or conf >= 0.55

    def reset_active_green_confirmation(self):
        self.tl_active_green_count = 0
        self.tl_active_green_since = 0.0
        self.tl_active_green_last_time = 0.0
        self.tl_active_green_stopline_id = None

    def decision_matches_stopline_id(self, line_id, data=None):
        line_id = str(line_id or "")
        if not line_id:
            return False
        data = data if isinstance(data, dict) else (getattr(self, "latest_decision", {}) or {})
        ids = []

        def add_id(value):
            value = str(value or "").strip()
            if value:
                ids.append(value)

        for key in (
            "selected_stopline_id",
            "active_tl_stopline_id",
            "stopline_id",
            "traffic_light_stopline_id",
            "traffic_light_id",
            "tl_id",
        ):
            add_id(data.get(key))

        tl = data.get("traffic_light") if isinstance(data, dict) else None
        if isinstance(tl, dict):
            for key in (
                "selected_stopline_id",
                "stopline_id",
                "traffic_light_stopline_id",
                "traffic_light_id",
                "tl_id",
                "id",
            ):
                add_id(tl.get(key))

        if not ids:
            return True

        def id_tokens(value):
            tokens = []
            part = ""
            for ch in str(value or ""):
                if ch.isalnum():
                    part += ch
                elif part:
                    tokens.append(part)
                    part = ""
            if part:
                tokens.append(part)
            return tokens

        line_tokens = set(id_tokens(line_id))
        for candidate in ids:
            candidate_tokens = set(id_tokens(candidate))
            if (
                candidate == line_id
                or (bool(candidate_tokens) and candidate_tokens.issubset(line_tokens))
                or (bool(line_tokens) and line_tokens.issubset(candidate_tokens))
            ):
                return True
        return False

    def active_stopline_verified_green_release_ready(self, line_id, metrics, reason):
        data = getattr(self, "latest_decision", {}) or {}
        if not self.is_verified_green_decision(data, reason):
            self.reset_active_green_confirmation()
            return False
        if not self.decision_matches_stopline_id(line_id, data):
            self.reset_active_green_confirmation()
            return False
        if self.selected_stopline_passed_ignore():
            return True
        if not isinstance(metrics, dict) or bool(metrics.get("passed", False)):
            return True

        now = time.time()
        line_id = str(line_id or "")
        last_id = str(getattr(self, "tl_active_green_stopline_id", "") or "")
        last_time = float(getattr(self, "tl_active_green_last_time", 0.0) or 0.0)

        if last_id != line_id or last_time <= 0.0 or now - last_time > 0.60:
            self.tl_active_green_stopline_id = line_id
            self.tl_active_green_since = now
            self.tl_active_green_count = 1
        else:
            self.tl_active_green_count = int(getattr(self, "tl_active_green_count", 0) or 0) + 1
        self.tl_active_green_last_time = now

        since = float(getattr(self, "tl_active_green_since", 0.0) or 0.0)
        age = now - since if since > 0.0 else 0.0
        return int(getattr(self, "tl_active_green_count", 0) or 0) >= 2 and age >= 0.08

    def active_stopline_green_detected_release_ready(self, line_id, metrics, reason):
        if not bool(getattr(self, "red_hold_active", False)):
            return False
        if not line_id or not isinstance(metrics, dict):
            return False
        try:
            speed_now = float(self.get_speed())
            control_dist = float(metrics.get("control_dist"))
        except Exception:
            return False
        if speed_now >= 0.3 or control_dist < -2.5 or control_dist > 3.0:
            return False

        latest = getattr(self, "latest_decision", {}) or {}
        text = (str(reason or "") + "|" + str(latest.get("reason", "") or "")).lower()
        state = self.latest_tl_state()
        green_signal = (
            state == "green"
            or "green_light_weak_not_latched_release" in text
            or "green_light_controlled_release" in text
            or "green_light_confirmed_stable" in text
        )
        last_green_age = time.time() - float(getattr(self, "tl_last_green_seen_time", 0.0) or 0.0)
        recent_green = last_green_age <= 0.8 and int(getattr(self, "tl_green_seen_count", 0) or 0) >= 1
        if not (green_signal or recent_green):
            return False
        return self.decision_matches_stopline_id(line_id, latest)

    def get_stopline_by_id(self, line_id):
        if not line_id:
            return None
        active_line = getattr(self, "active_tl_stopline_line", None)
        if isinstance(active_line, dict) and str(active_line.get("id", "")) == str(line_id):
            return active_line
        for line in getattr(self, "manual_tl_stoplines", []) or []:
            if str(line.get("id", "")) == str(line_id):
                return line
        return None

    def stopline_metrics(self, line):
        try:
            ego_tf = self.ego.get_transform()
            fwd = ego_tf.get_forward_vector()
            right = ego_tf.get_right_vector()
            front_x, front_y = self.get_ego_front_xy()
            dx = float(line["x"]) - float(front_x)
            dy = float(line["y"]) - float(front_y)
            line_dist = dx * float(fwd.x) + dy * float(fwd.y)
            lat = dx * float(right.x) + dy * float(right.y)
            stop_before = float(line.get("stop_before_m", self.tl_stopline_stop_before_m))
            control_dist = float(line_dist) - stop_before
            passed = line_dist < 0.30 or control_dist < 0.0
            return {
                "line_dist": float(line_dist),
                "lat": float(lat),
                "stop_before": float(stop_before),
                "control_dist": float(control_dist),
                "passed": bool(passed),
            }
        except Exception:
            return None

    def update_stopline_debug(self, line_id=None, metrics=None, reason=""):
        metrics = metrics or {}
        has_line = line_id is not None
        self.last_tl_stopline_debug = {
            "selected_stopline_id": line_id,
            "line_dist": round(float(metrics.get("line_dist", 0.0)), 3)
            if has_line and metrics.get("line_dist") is not None else None,
            "control_dist": round(float(metrics.get("control_dist", 0.0)), 3)
            if has_line and metrics.get("control_dist") is not None else None,
            "stop_before": round(float(metrics.get("stop_before", self.tl_stopline_stop_before_m)), 3)
            if has_line and metrics.get("stop_before") is not None else None,
            "lat": round(float(metrics.get("lat", 0.0)), 3)
            if has_line and metrics.get("lat") is not None else None,
            "red_hold_active": bool(getattr(self, "red_hold_active", False)),
            "passed_ignore": self.selected_stopline_passed_ignore(),
            "tl_state": self.latest_tl_state(),
            "reason": str(reason or ""),
        }

    def apply_no_stopline_tl_fallback(self, target_speed, reason):
        now = time.time()
        text = str(reason or "").lower()
        latest = getattr(self, "latest_decision", {}) or {}
        decision = str(latest.get("decision", "") or "").upper()
        tl_state = self.latest_tl_state()

        red_or_yellow = (
            tl_state in {"red", "yellow"}
            or "red_light" in text
            or "yellow_light" in text
            or bool(getattr(self, "red_hold_active", False))
        )
        if not red_or_yellow:
            return target_speed, reason

        self.red_hold_active = False
        self.red_hold_until_green = False
        self.tl_stop_hold_active = False
        self.tl_stop_hold_until = 0.0
        self.active_tl_stopline_id = None
        self.active_tl_stopline_line = None
        self.active_tl_stopline_line_dist = None
        self.active_tl_stopline_control_dist = None
        self.active_tl_stopline_lat = None
        self.no_stopline_red_hold_since = 0.0
        self.no_stopline_last_red_yellow_time = 0.0

        self.update_stopline_debug(None, {}, reason)
        self.last_tl_stopline_debug["passed_ignore"] = False

        capped = min(float(target_speed or 0.0), 1.5)
        if capped <= 0.05:
            capped = 1.0
        if decision == "STOP" or "driver_only_decision_stop" in text or "overhead_stop" in text or "no_sensor_stop" in text:
            capped = max(capped, 1.0)
        return capped, f"{reason}|tl_red_no_stopline_no_full_stop"

    def no_stopline_hold_release_speed(self, target_speed):
        try:
            base = float(target_speed or 0.0)
        except Exception:
            base = 0.0
        if base <= 0.05:
            base = float(getattr(self, "green_release_min_start_speed_mps", 2.0) or 2.0)
        return self.clamp(base, 1.2, 3.0)

    def maybe_release_no_stopline_hold(self, target_speed, reason):
        if not bool(getattr(self, "red_hold_active", False)):
            return None
        if bool(getattr(self, "active_tl_stopline_id", None)):
            return None

        now = time.time()
        latest = getattr(self, "latest_decision", {}) or {}
        latest_reason = str(latest.get("reason", "") or "").lower()
        tl_state = self.latest_tl_state()

        green_release = self.is_verified_green_decision(latest, latest_reason)

        if green_release:
            speed = self.no_stopline_hold_release_speed(target_speed)
            self.clear_red_hold(mark_green_release=True)
            return speed, f"{reason}|tl_red_hold_release_verified_green|tl_green_release_no_stopline_hold"

        red_yellow_now = self.current_tl_red_yellow(reason)
        if red_yellow_now:
            self.no_stopline_last_red_yellow_time = now
            return None

        decision_go = str(latest.get("decision", "") or "").upper() == "GO"
        if decision_go:
            if float(getattr(self, "no_stopline_decision_go_since", 0.0) or 0.0) <= 0.0:
                self.no_stopline_decision_go_since = now
        else:
            self.no_stopline_decision_go_since = 0.0
            return None

        hold_since = float(getattr(self, "no_stopline_red_hold_since", 0.0) or 0.0)
        last_red = float(getattr(self, "no_stopline_last_red_yellow_time", 0.0) or 0.0)
        go_since = float(getattr(self, "no_stopline_decision_go_since", 0.0) or 0.0)

        hold_age = now - hold_since if hold_since > 0.0 else 0.0
        clear_age = now - last_red if last_red > 0.0 else 999.0
        go_age = now - go_since if go_since > 0.0 else 0.0

        try:
            speed_now = float(self.get_speed())
        except Exception:
            speed_now = 0.0

        if hold_age >= 2.0 and clear_age >= 1.2 and go_age >= 1.0 and speed_now <= 0.35:
            return (
                0.0,
                f"{reason}|tl_red_hold_keep_on_unknown:"
                f"no_red_yellow_clear_sec={clear_age:.1f},"
                f"decision_go_stable_sec={go_age:.1f}",
            )

        return None

    def make_virtual_tl_stopline(self):
        try:
            ego_tf = self.ego.get_transform()
            ego_loc = ego_tf.location
            fwd = ego_tf.get_forward_vector()
            right = ego_tf.get_right_vector()
            front_x, front_y = self.get_ego_front_xy()
            ego_wp = self.map.get_waypoint(
                ego_loc,
                project_to_road=True,
                lane_type=self.carla.LaneType.Driving,
            )
        except Exception:
            return None

        if ego_wp is None:
            return None

        def _line_at(loc, wp, suffix, source):
            dx = float(loc.x) - float(front_x)
            dy = float(loc.y) - float(front_y)
            line_dist = dx * float(fwd.x) + dy * float(fwd.y)
            lat = dx * float(right.x) + dy * float(right.y)
            road_id = getattr(wp, "road_id", None)
            lane_id = getattr(wp, "lane_id", None)
            line = {
                "id": f"virtual_stopline_{suffix}_road_{road_id}_lane_{lane_id}",
                "source": source,
                "map": str(getattr(self.map, "name", "")).split("/")[-1],
                "x": float(loc.x),
                "y": float(loc.y),
                "yaw_deg": float(getattr(wp.transform.rotation, "yaw", ego_tf.rotation.yaw)),
                "road_id": road_id,
                "lane_id": lane_id,
                "approach_m": 80.0,
                "crawl_m": 14.0,
                "release_after_pass_m": 0.25,
                "lateral_half_width_m": 5.0,
                "approach_speed_mps": 4.0,
                "crawl_speed_mps": 1.0,
                "stop_speed_mps": 0.0,
                "stop_before_m": float(self.tl_stopline_stop_before_m),
                "virtual": True,
            }
            return line, float(line_dist), float(lat)

        if bool(getattr(ego_wp, "is_junction", False)):
            loc = ego_tf.location
            hit = _line_at(loc, ego_wp, "passed_junction", "virtual_junction_passed")
            line, _, lat = hit
            return line, -0.5, lat

        wp = ego_wp
        prev_wp = ego_wp
        max_scan_m = 90.0
        step_m = 1.0
        travelled = 0.0

        while travelled <= max_scan_m:
            try:
                next_wps = wp.next(step_m)
            except Exception:
                next_wps = []
            if not next_wps:
                break

            def _score_next(cand):
                try:
                    cf = cand.transform.get_forward_vector()
                    return float(cf.x) * float(fwd.x) + float(cf.y) * float(fwd.y)
                except Exception:
                    return 0.0

            prev_wp = wp
            wp = max(next_wps, key=_score_next)
            travelled += step_m

            if bool(getattr(wp, "is_junction", False)):
                loc = prev_wp.transform.location
                hit = _line_at(loc, prev_wp, "junction_entry", "virtual_junction_entry")
                line, line_dist, lat = hit
                if abs(float(lat)) > 4.0:
                    return None
                return line, line_dist, lat

        return None

    def nearest_manual_tl_stopline(self):
        lines = getattr(self, "manual_tl_stoplines", []) or []
        self._last_manual_tl_candidates_debug = ""
        if not lines:
            return None

        map_name = ""
        try:
            map_name = str(self.map.name).split("/")[-1]
        except Exception:
            map_name = ""

        try:
            ego_tf = self.ego.get_transform()
            ego_loc = ego_tf.location
            fwd = ego_tf.get_forward_vector()
            right = ego_tf.get_right_vector()
            ego_yaw = float(ego_tf.rotation.yaw)
        except Exception:
            return None

        try:
            front_x, front_y = self.get_ego_front_xy()
        except Exception:
            front_x, front_y = float(ego_loc.x), float(ego_loc.y)

        try:
            ego_wp = self.map.get_waypoint(
                ego_loc,
                project_to_road=True,
                lane_type=self.carla.LaneType.Driving,
            )
        except Exception:
            ego_wp = None

        ego_road = getattr(ego_wp, "road_id", None) if ego_wp is not None else None
        ego_lane = getattr(ego_wp, "lane_id", None) if ego_wp is not None else None

        candidates = []

        def _angle_diff(a, b):
            try:
                return abs(self.angle_norm_deg(float(a) - float(b)))
            except Exception:
                d = (float(a) - float(b) + 180.0) % 360.0 - 180.0
                return abs(d)

        for line in lines:
            line_map = str(line.get("map", "") or "").split("/")[-1]
            if line_map and map_name and line_map != map_name:
                continue

            try:
                lx = float(line["x"])
                ly = float(line["y"])
            except Exception:
                continue

            dx = lx - front_x
            dy = ly - front_y

            longitudinal_m = dx * float(fwd.x) + dy * float(fwd.y)
            lateral_m = dx * float(right.x) + dy * float(right.y)

            release_after = float(line.get("release_after_pass_m", 4.0))
            approach_m = float(line.get("approach_m", 120.0))
            line_id = str(line.get("id", "manual_stopline"))
            line_road = line.get("road_id", None)
            line_lane = line.get("lane_id", None)
            yaw_diff = _angle_diff(line.get("yaw_deg", 0.0), ego_yaw)

            same_road = False
            same_lane = False
            try:
                same_road = (
                    ego_road is not None
                    and line_road is not None
                    and int(line_road) == int(ego_road)
                )
                same_lane = (
                    same_road
                    and ego_lane is not None
                    and line_lane is not None
                    and int(line_lane) == int(ego_lane)
                )
            except Exception:
                same_road = False
                same_lane = False

            abs_lat = abs(float(lateral_m))
            if abs_lat > 4.0:
                continue
            if abs_lat > 3.0 and not (same_lane or (same_road and yaw_diff <= 35.0)):
                continue

            recent_passed = getattr(self, "tl_recently_passed_stoplines", {}) or {}
            if time.time() < float(recent_passed.get(line_id, 0.0) or 0.0):
                continue

            stop_before_m = float(line.get("stop_before_m", self.tl_stopline_stop_before_m))
            control_dist_m = longitudinal_m - stop_before_m

            if longitudinal_m < 0.30 or control_dist_m < -max(0.25, release_after):
                self.mark_stopline_passed(line_id)
                continue
            if longitudinal_m > approach_m:
                continue

            candidates.append({
                "line": line,
                "dist": float(longitudinal_m),
                "lat": float(lateral_m),
                "yaw_diff": float(yaw_diff),
                "same_road": bool(same_road),
                "same_lane": bool(same_lane),
            })

        if not candidates:
            virtual_hit = self.make_virtual_tl_stopline()
            if virtual_hit is not None:
                line, dist, lat = virtual_hit
                self._last_manual_tl_candidates_debug = (
                    f"{str(line.get('id','virtual'))}:d={float(dist):.1f},"
                    f"lat={float(lat):.1f},virtual=1"
                )
                return virtual_hit
            self._last_manual_tl_candidates_debug = "none_in_front"
            return None

        # Debug için en yakın adayları sakla.
        try:
            tops = sorted(
                candidates,
                key=lambda c: abs(c["lat"]) + 0.03 * max(0.0, c["dist"]) + 0.003 * c["yaw_diff"]
            )[:5]
            self._last_manual_tl_candidates_debug = ";".join(
                f"{str(c['line'].get('id','?'))}:d={c['dist']:.1f},lat={c['lat']:.1f},yaw={c['yaw_diff']:.0f},same={int(c['same_lane'])}"
                for c in tops
            )
        except Exception:
            self._last_manual_tl_candidates_debug = "debug_failed"

        strict = []
        for c in candidates:
            if abs(c["lat"]) <= 2.5 and (c["same_lane"] or c["yaw_diff"] <= 55.0):
                score = abs(c["lat"]) * 3.0 + c["dist"] * 0.020 + c["yaw_diff"] * 0.010
                if c["same_lane"]:
                    score -= 10.0
                strict.append((score, c))

        if strict:
            _, c = min(strict, key=lambda x: x[0])
            return c["line"], c["dist"], c["lat"]

        loose = []
        for c in candidates:
            if abs(c["lat"]) <= 4.0 and (c["same_lane"] or (c["same_road"] and c["yaw_diff"] <= 35.0)):
                score = abs(c["lat"]) * 5.0 + c["dist"] * 0.030 + c["yaw_diff"] * 0.015
                if c["same_lane"]:
                    score -= 8.0
                elif c["same_road"]:
                    score -= 3.0
                loose.append((score, c))

        if loose:
            _, c = min(loose, key=lambda x: x[0])
            return c["line"], c["dist"], c["lat"]

        virtual_hit = self.make_virtual_tl_stopline()
        if virtual_hit is not None:
            line, dist, lat = virtual_hit
            self._last_manual_tl_candidates_debug = (
                f"{str(line.get('id','virtual'))}:d={float(dist):.1f},"
                f"lat={float(lat):.1f},virtual=1"
            )
            return virtual_hit

        return None


    def apply_manual_tl_stopline_cap(self, target_speed, reason):
        if not getattr(self, "manual_tl_stoplines_enabled", False):
            return target_speed, reason

        stage = str(self.get_stage() or "")
        if stage == "PASSENGER_STOP":
            return target_speed, reason

        reason_l = str(reason or "").lower()
        latest_reason_l = str((getattr(self, "latest_decision", {}) or {}).get("reason", "") or "").lower()
        latest_state = self.latest_tl_state()
        green_release_text = (
            "green_light_weak_not_latched_release" in reason_l
            or "green_light_weak_not_latched_release" in latest_reason_l
            or "green_light_controlled_release" in reason_l
            or "green_light_controlled_release" in latest_reason_l
            or "green_light_confirmed_stable" in reason_l
            or "green_light_confirmed_stable" in latest_reason_l
            or latest_state == "green"
            or (
                time.time() - float(getattr(self, "tl_last_green_seen_time", 0.0) or 0.0) <= 0.8
                and int(getattr(self, "tl_green_seen_count", 0) or 0) >= 1
            )
        )
        current_red_yellow = (
            latest_state in {"red", "yellow"}
            or "red_light" in reason_l
            or "yellow_light" in reason_l
            or "traffic_light_red" in reason_l
            or "traffic_light_yellow" in reason_l
            or "red_light" in latest_reason_l
            or "yellow_light" in latest_reason_l
            or "traffic_light_red" in latest_reason_l
            or "traffic_light_yellow" in latest_reason_l
        )
        recent_green_blocks_red = time.time() < float(
            getattr(self, "tl_recent_green_blocks_red_until", 0.0) or 0.0
        )
        if recent_green_blocks_red and current_red_yellow:
            reason = f"{reason}|tl_red_yellow_blocked_by_recent_green"
            reason_l = str(reason or "").lower()
            current_red_yellow = False
        current_red_yellow = (
            current_red_yellow
            and latest_state != "green"
            and not green_release_text
        )
        reliable_current_red_yellow = current_red_yellow
        rejected_or_unknown_tl = (
            latest_state == "unknown"
            or "tl_pipeline_rejected" in reason_l
            or "pipeline_rejected" in reason_l
            or "outside_active_roi" in reason_l
            or "tl_pipeline_rejected" in latest_reason_l
            or "pipeline_rejected" in latest_reason_l
            or "outside_active_roi" in latest_reason_l
        )
        yellow_control = (
            "yellow_light" in reason_l
            or "traffic_light_yellow" in reason_l
        )
        yellow_control = yellow_control and current_red_yellow
        active_stopline_id = str(getattr(self, "active_tl_stopline_id", "") or "")
        green_confirmed = self.is_verified_green_decision(reason=reason)
        last_reliable_state = str(getattr(self, "tl_last_reliable_state", "") or "").lower()
        red_yellow_memory_active = (
            bool(active_stopline_id)
            and bool(getattr(self, "red_hold_active", False))
            and last_reliable_state in {"red", "yellow"}
        )
        if red_yellow_memory_active and not green_confirmed and not green_release_text:
            green_release_text = False
            current_red_yellow = True

        if red_yellow_memory_active and green_release_text:
            release_id = active_stopline_id
            self.release_wait_green_latch()
            self.clear_red_hold(mark_green_release=True)
            return max(float(target_speed or 0.0), 1.2), (
                f"{reason}|tl_green_release_before_red_memory"
                f"|tl_green_blocks_red_memory_set"
                f"|tl_red_hold_release_on_green_detected"
                f"|tl_wait_green_latch_release_verified_green:id={release_id}"
            )

        if green_confirmed and not active_stopline_id:
            self.clear_red_hold(mark_green_release=True)
            return max(float(target_speed or 0.0), 1.2), (
                f"{reason}|tl_red_hold_release_verified_green"
            )

        hit = None
        released_bad_lateral = False
        hold_active = bool(getattr(self, "red_hold_active", False)) and bool(
            getattr(self, "active_tl_stopline_id", None)
        )

        if hold_active:
            line = self.get_stopline_by_id(self.active_tl_stopline_id)
            metrics = self.stopline_metrics(line) if line is not None else None
            if metrics is None:
                self.update_stopline_debug(self.active_tl_stopline_id, {}, reason)
                return 0.0, f"{reason}|tl_red_hold_unknown_wait:no_stopline_metrics"

            try:
                hold_speed = float(self.get_speed())
            except Exception:
                hold_speed = 0.0

            stopped_wait_candidate = (
                hold_speed < 0.3
                and str(getattr(self, "tl_last_reliable_state", "") or "").lower() in {"red", "yellow"}
                and float(metrics.get("control_dist", 0.0)) >= -4.0
                and float(metrics.get("control_dist", 0.0)) <= 3.0
                and float(metrics.get("line_dist", 0.0)) > -4.0
            )
            if (
                stopped_wait_candidate
                and not bool(getattr(self, "tl_waiting_for_verified_green", False))
                and not self.tl_wait_green_release_allowed(reason)
            ):
                self.set_wait_green_latch(self.active_tl_stopline_id, metrics, reason)
                return 0.0, (
                    self.wait_green_latch_reason(
                        f"{reason}|tl_wait_green_latch_set_after_red_stop",
                        metrics,
                    )
                )

            if self.active_stopline_green_detected_release_ready(
                self.active_tl_stopline_id,
                metrics,
                reason,
            ):
                release_id = self.active_tl_stopline_id
                last_green_age = time.time() - float(getattr(self, "tl_last_green_seen_time", 0.0) or 0.0)
                self.update_stopline_debug(release_id, metrics, reason)
                self.release_wait_green_latch()
                self.clear_red_hold(mark_green_release=True)
                speed = max(float(target_speed or 0.0), float(getattr(self, "green_release_min_start_speed_mps", 2.0) or 2.0))
                return speed, (
                    f"{reason}|tl_red_hold_release_on_green_detected:"
                    f"id={release_id},current_tl_state={latest_state},"
                    f"decision_reason={latest_reason_l},last_green_seen_age={last_green_age:.2f},"
                    f"green_release_allowed=True,red_hold_blocked_by_green=True,"
                    f"release_block_reason=none|{self.tl_wait_green_release_reason_tag(reason)}"
                )

            if bool(getattr(self, "tl_waiting_for_verified_green", False)) and not self.tl_wait_green_release_allowed(reason):
                self.set_wait_green_latch(self.active_tl_stopline_id, metrics, reason)
                return 0.0, self.wait_green_latch_reason(reason, metrics)

            if metrics["passed"]:
                far_passed = (
                    str(self.active_tl_stopline_id).startswith("virtual_stopline_passed")
                    or float(metrics.get("control_dist", 0.0)) <= -5.0
                    or float(metrics.get("line_dist", 0.0)) <= -3.0
                )
                if far_passed:
                    passed_id = self.active_tl_stopline_id
                    self.mark_stopline_passed(passed_id)
                    self.update_stopline_debug(passed_id, metrics, reason)
                    self.last_tl_stopline_debug["passed_ignore"] = True
                    passed_speed = self.clamp(
                        max(
                            float(target_speed or 0.0),
                            float(getattr(self, "go_speed_mps", 0.0) or 0.0),
                            1.2,
                        ),
                        0.0,
                        self.max_speed_mps,
                    )
                    return passed_speed, (
                        f"{reason}|tl_ignore_passed_virtual_stopline"
                        f"|tl_ignore_decision_slow_for_passed_stopline"
                        f"|tl_overshoot_recovery_blocked_passed_stopline:"
                        f"id={passed_id},line_dist={metrics['line_dist']:.2f},"
                        f"control_dist={metrics['control_dist']:.2f},lat={metrics['lat']:.2f},"
                        f"v={hold_speed:.2f}"
                    )
                stopped_after_overshoot = (
                    reliable_current_red_yellow
                    and hold_speed < 0.3
                    and float(metrics.get("control_dist", 0.0)) >= -4.0
                    and float(metrics.get("control_dist", 0.0)) <= 0.5
                    and float(metrics.get("line_dist", 0.0)) > -3.0
                )
                if stopped_after_overshoot:
                    self.set_wait_green_latch(self.active_tl_stopline_id, metrics, reason)
                    return 0.0, (
                        f"{reason}|tl_crossed_release_suppressed_red_active"
                        f"|tl_red_hold_stopped_after_overshoot"
                        f"|tl_wait_green_latch_set_after_red_stop"
                        f"|tl_wait_green_latch_hold"
                        f"|tl_stationary_hold_no_rollback:"
                        f"id={self.active_tl_stopline_id},line_dist={metrics['line_dist']:.2f},"
                        f"control_dist={metrics['control_dist']:.2f},lat={metrics['lat']:.2f},"
                        f"v={hold_speed:.2f}"
                    )
                if (
                    reliable_current_red_yellow
                    and (
                        time.time() < float(getattr(self, "tl_overshoot_recover_until", 0.0) or 0.0)
                        or (
                            hold_speed > 0.3
                            and float(metrics.get("control_dist", 0.0)) > -5.0
                            and float(metrics.get("line_dist", 0.0)) > -3.0
                        )
                    )
                ):
                    self.update_stopline_debug(self.active_tl_stopline_id, metrics, reason)
                    self.red_hold_active = True
                    self.red_hold_until_green = True
                    self.tl_stop_hold_active = True
                    self.tl_stop_hold_until = time.time() + 8.0
                    if float(getattr(self, "tl_overshoot_recover_until", 0.0) or 0.0) < time.time():
                        self.tl_overshoot_recover_until = time.time() + 1.0
                    self.last_tl_stopline_debug["passed_ignore"] = False
                    return 0.0, (
                        f"{reason}|tl_red_overshoot_recover_stop:"
                        f"id={self.active_tl_stopline_id},line_dist={metrics['line_dist']:.2f},"
                        f"control_dist={metrics['control_dist']:.2f},lat={metrics['lat']:.2f},"
                        f"v={hold_speed:.2f}"
                    )
                if reliable_current_red_yellow and not (
                    hold_speed > 1.0
                    and (
                        float(metrics.get("control_dist", 0.0)) <= -5.0
                        or float(metrics.get("line_dist", 0.0)) <= -3.0
                    )
                ):
                    if hold_speed < 0.3:
                        self.set_wait_green_latch(self.active_tl_stopline_id, metrics, reason)
                        return 0.0, self.wait_green_latch_reason(
                            f"{reason}|tl_crossed_release_suppressed_red_active"
                            f"|tl_red_overshoot_hold_until_green"
                            f"|tl_wait_green_latch_set_after_red_stop",
                            metrics,
                        )
                    self.update_stopline_debug(self.active_tl_stopline_id, metrics, reason)
                    self.red_hold_active = True
                    self.red_hold_until_green = True
                    self.tl_stop_hold_active = True
                    self.tl_stop_hold_until = time.time() + 8.0
                    self.last_tl_stopline_debug["passed_ignore"] = False
                    return 0.0, (
                        f"{reason}|tl_crossed_release_suppressed_red_active"
                        f"|tl_red_overshoot_hold_until_green:"
                        f"id={self.active_tl_stopline_id},line_dist={metrics['line_dist']:.2f},"
                        f"control_dist={metrics['control_dist']:.2f},lat={metrics['lat']:.2f},"
                        f"v={hold_speed:.2f}"
                    )
                crossed_id = self.active_tl_stopline_id
                self.mark_stopline_passed(crossed_id)
                self.update_stopline_debug(crossed_id, metrics, reason)
                self.last_tl_stopline_debug["passed_ignore"] = True
                return max(float(target_speed or 0.0), 1.2), (
                    f"{reason}|tl_stopline_crossed_release_hold:"
                    f"id={crossed_id},line_dist={metrics['line_dist']:.2f},"
                    f"control_dist={metrics['control_dist']:.2f},lat={metrics['lat']:.2f}"
                )

            if abs(float(metrics.get("lat", 999.0))) > 4.0:
                bad_id = self.active_tl_stopline_id
                self.update_stopline_debug(bad_id, metrics, reason)
                self.clear_red_hold()
                reason = (
                    f"{reason}|tl_red_hold_released_bad_lateral:"
                    f"id={bad_id},line_dist={metrics['line_dist']:.2f},"
                    f"control_dist={metrics['control_dist']:.2f},lat={metrics['lat']:.2f}"
                )
                reason_l = str(reason or "").lower()
                hold_active = False
                released_bad_lateral = True
                if not current_red_yellow:
                    return target_speed, reason
            if hold_active:
                self.active_tl_stopline_line_dist = metrics["line_dist"]
                self.active_tl_stopline_control_dist = metrics["control_dist"]
                self.active_tl_stopline_lat = metrics["lat"]

                if self.active_stopline_verified_green_release_ready(
                    self.active_tl_stopline_id,
                    metrics,
                    reason,
                ):
                    release_id = self.active_tl_stopline_id
                    self.update_stopline_debug(release_id, metrics, reason)
                    self.release_wait_green_latch()
                    self.clear_red_hold(mark_green_release=True)
                    return max(float(target_speed or 0.0), 1.2), (
                        f"{reason}|tl_red_hold_release_verified_green"
                        f"|tl_wait_green_latch_release_verified_green:"
                        f"id={release_id},line_dist={metrics['line_dist']:.2f},"
                        f"control_dist={metrics['control_dist']:.2f},lat={metrics['lat']:.2f}"
                    )

                hold_tag = "tl_red_hold_until_verified_green"
                if (
                    latest_state == "unknown"
                    or "tl_pipeline_rejected" in reason_l
                    or "pipeline_rejected" in reason_l
                ):
                    hold_tag = "tl_red_hold_keep_on_unknown_memory"
                reason = f"{reason}|{hold_tag}"
                if "driver_only_decision_go" in reason_l:
                    reason = f"{reason}|tl_red_memory_overrides_decision_go"
                reason_l = str(reason or "").lower()
                current_red_yellow = current_red_yellow or not green_release_text
                yellow_control = yellow_control and "keep_on_unknown" not in hold_tag
                hit = (line, metrics["line_dist"], metrics["lat"])

        if hit is None and not self.manual_tl_red_active() and not current_red_yellow:
            if green_release_text:
                return max(float(target_speed or 0.0), 1.2), (
                    f"{reason}|tl_green_blocks_red_memory_set"
                    f"|tl_red_memory_set_blocked_reason=green_evidence"
                )
            if bool(getattr(self, "red_hold_active", False)):
                return self.apply_no_stopline_tl_fallback(
                    target_speed,
                    f"{reason}|tl_red_hold_unknown_wait",
                )
            return target_speed, reason

        if hit is None:
            hit = self.nearest_manual_tl_stopline()

        if hit is None:
            dbg = str(getattr(self, "_last_manual_tl_candidates_debug", "") or "no_debug")
            if released_bad_lateral:
                return target_speed, reason
            if current_red_yellow or bool(getattr(self, "red_hold_active", False)):
                return self.apply_no_stopline_tl_fallback(
                    target_speed,
                    f"{reason}|manual_tl_stopline_no_match:{dbg}",
                )

            if self.stopline_recently_passed():
                self.tl_stop_hold_active = False
                self.tl_stop_hold_until = 0.0
                return max(float(target_speed or 0.0), 1.2), (
                    f"{reason}|tl_stopline_recently_passed_ignore:{dbg}"
                )
            return target_speed, f"{reason}|manual_tl_stopline_no_match:{dbg}"

        line, dist_m, lat_m = hit
        line_id = str(line.get("id", "manual_stopline"))
        preliminary_stop_before_m = float(line.get("stop_before_m", self.tl_stopline_stop_before_m))
        preliminary_control_dist_m = float(dist_m) - preliminary_stop_before_m
        preliminary_metrics = {
            "line_dist": dist_m,
            "control_dist": preliminary_control_dist_m,
            "stop_before": preliminary_stop_before_m,
            "lat": lat_m,
        }
        try:
            preliminary_speed = float(self.get_speed())
        except Exception:
            preliminary_speed = 0.0
        if (
            bool(getattr(self, "tl_waiting_for_verified_green", False))
            and preliminary_speed < 0.3
            and not self.tl_wait_green_release_allowed(reason)
        ):
            self.set_wait_green_latch(
                str(getattr(self, "tl_wait_stopline_id", "") or line_id),
                preliminary_metrics,
                reason,
            )
            return 0.0, self.wait_green_latch_reason(reason, preliminary_metrics)

        cache_hit = self.stopline_recently_passed(line_id) or self.stopline_recently_released(line_id)
        cache_near_or_passed = (
            str(line_id).startswith("virtual_stopline_passed")
            or float(dist_m) <= 3.0
            or preliminary_control_dist_m <= 0.5
        )
        if cache_hit and cache_near_or_passed:
            self.mark_stopline_passed(line_id)
            self.update_stopline_debug(
                line_id,
                {
                    "line_dist": dist_m,
                    "control_dist": preliminary_control_dist_m,
                    "stop_before": preliminary_stop_before_m,
                    "lat": lat_m,
                },
                f"{reason}|tl_released_stopline_cache_hit|tl_ignore_reacquired_passed_stopline",
            )
            self.last_tl_stopline_debug["passed_ignore"] = True
            return max(float(target_speed or 0.0), 1.2), (
                f"{reason}|tl_released_stopline_cache_hit"
                f"|tl_ignore_reacquired_passed_stopline:"
                f"id={line_id},line_dist={float(dist_m):.2f},"
                f"control_dist={preliminary_control_dist_m:.2f},lat={float(lat_m):.2f}"
            )

        hard_passed_stopline = (
            str(line_id).startswith("virtual_stopline_passed")
            or float(dist_m) <= -3.0
            or preliminary_control_dist_m <= -5.0
        )
        if hard_passed_stopline:
            self.mark_stopline_passed(line_id)
            self.update_stopline_debug(
                line_id,
                {
                    "line_dist": dist_m,
                    "control_dist": preliminary_control_dist_m,
                    "stop_before": preliminary_stop_before_m,
                    "lat": lat_m,
                },
                f"{reason}|tl_ignore_passed_virtual_stopline",
            )
            self.last_tl_stopline_debug["passed_ignore"] = True
            passed_speed = self.clamp(
                max(
                    float(target_speed or 0.0),
                    float(getattr(self, "go_speed_mps", 0.0) or 0.0),
                    1.2,
                ),
                0.0,
                self.max_speed_mps,
            )
            return passed_speed, (
                f"{reason}|tl_ignore_passed_virtual_stopline"
                f"|tl_ignore_decision_slow_for_passed_stopline"
                f"|tl_overshoot_recovery_blocked_passed_stopline:"
                f"id={line_id},line_dist={float(dist_m):.2f},"
                f"control_dist={preliminary_control_dist_m:.2f},lat={float(lat_m):.2f}"
            )

        if abs(float(lat_m)) > 4.0:
            self.update_stopline_debug(
                line_id,
                {
                    "line_dist": dist_m,
                    "control_dist": float(dist_m) - float(line.get("stop_before_m", self.tl_stopline_stop_before_m)),
                    "stop_before": float(line.get("stop_before_m", self.tl_stopline_stop_before_m)),
                    "lat": lat_m,
                },
                f"{reason}|tl_stopline_rejected_lateral",
            )
            return target_speed, (
                f"{reason}|tl_stopline_rejected_lateral:"
                f"id={line_id},line_dist={float(dist_m):.2f},lat={float(lat_m):.2f}"
            )

        if current_red_yellow:
            self.set_active_tl_stopline(line, dist_m, lat_m)
            state_tag = "tl_yellow_hold_memory_set" if latest_state == "yellow" else "tl_red_hold_memory_set"
            if state_tag not in reason_l:
                reason = f"{reason}|tl_red_memory_set_allowed|{state_tag}"
                reason_l = str(reason or "").lower()

        stop_before_m = float(line.get("stop_before_m", self.tl_stopline_stop_before_m))
        control_dist_m = float(dist_m) - stop_before_m
        try:
            current_speed = float(self.get_speed())
        except Exception:
            current_speed = 0.0
        v = max(0.0, current_speed)
        front_bumper_offset_m = float(getattr(self, "tl_front_bumper_offset_m", 2.3) or 2.3)
        desired_fb_stop_before_m = float(
            getattr(self, "tl_desired_front_bumper_stop_before_m", 0.8) or 0.8
        )
        fb_line_dist_m = float(dist_m)
        fb_control_dist_m = fb_line_dist_m - desired_fb_stop_before_m
        center_line_dist_m = fb_line_dist_m + front_bumper_offset_m
        fb_reason = (
            f"line_dist_center={center_line_dist_m:.2f},"
            f"fb_line_dist={fb_line_dist_m:.2f},fb_control_dist={fb_control_dist_m:.2f},"
            f"front_bumper_offset={front_bumper_offset_m:.2f},"
            f"desired_fb_stop_before={desired_fb_stop_before_m:.2f}"
        )

        self.update_stopline_debug(
            line_id,
            {
                "line_dist": dist_m,
                "control_dist": control_dist_m,
                "stop_before": stop_before_m,
                "lat": lat_m,
            },
            reason,
        )
        memory_profile_active = (
            current_red_yellow
            and bool(getattr(self, "red_hold_active", False))
            and str(getattr(self, "active_tl_stopline_id", "") or "") == line_id
            and str(getattr(self, "tl_last_reliable_state", "") or "").lower() in {"red", "yellow"}
        )

        overshoot_recover_active = (
            reliable_current_red_yellow
            and fb_control_dist_m < -0.2
            and control_dist_m >= -4.0
            and dist_m > -3.0
            and (
                time.time() < float(getattr(self, "tl_overshoot_recover_until", 0.0) or 0.0)
                or v > 0.3
                or (v < 0.5 and control_dist_m >= -4.0 and control_dist_m <= 0.5 and dist_m > -3.0)
            )
        )
        if overshoot_recover_active:
            self.red_hold_active = True
            self.red_hold_until_green = True
            self.tl_stop_hold_active = True
            self.tl_stop_hold_until = time.time() + 8.0
            if float(getattr(self, "tl_overshoot_recover_until", 0.0) or 0.0) < time.time():
                self.tl_overshoot_recover_until = time.time() + 1.0
            self.last_tl_stopline_debug["passed_ignore"] = False
            recover_tag = "tl_red_overshoot_recover_hold"
            if v > 0.3:
                recover_tag = "tl_red_overshoot_recover_stop"
            return 0.0, (
                f"{reason}|{recover_tag}:"
                f"id={line_id},line_dist={dist_m:.2f},control_dist={control_dist_m:.2f},"
                f"lat={lat_m:.2f},v={v:.2f}"
            )

        far_passed_line = (
            str(line_id).startswith("virtual_stopline_passed")
            or dist_m <= -3.0
            or control_dist_m <= -5.0
        )
        if far_passed_line and (rejected_or_unknown_tl or not reliable_current_red_yellow):
            self.mark_stopline_passed(line_id)
            self.update_stopline_debug(
                line_id,
                {
                    "line_dist": dist_m,
                    "control_dist": control_dist_m,
                    "stop_before": stop_before_m,
                    "lat": lat_m,
                },
                f"{reason}|tl_passed_far_release_unknown|tl_passed_far_no_green_wait",
            )
            self.last_tl_stopline_debug["passed_ignore"] = True
            return max(float(target_speed or 0.0), 1.2), (
                f"{reason}|tl_passed_far_release_unknown|tl_passed_far_no_green_wait:"
                f"id={line_id},line_dist={dist_m:.2f},control_dist={control_dist_m:.2f},"
                f"lat={lat_m:.2f},v={v:.2f}"
            )

        if (
            reliable_current_red_yellow
            and not str(line_id).startswith("virtual_stopline_passed")
            and -4.0 <= control_dist_m <= 0.0
            and dist_m > -3.0
            and v <= 1.0
        ):
            self.red_hold_active = True
            self.red_hold_until_green = True
            self.tl_stop_hold_active = True
            self.tl_stop_hold_until = time.time() + 8.0
            self.last_tl_stopline_debug["passed_ignore"] = False
            return 0.0, (
                f"{reason}|tl_crossed_release_suppressed_red_active"
                f"|tl_red_overshoot_hold_until_green:"
                f"id={line_id},line_dist={dist_m:.2f},control_dist={control_dist_m:.2f},"
                f"lat={lat_m:.2f},v={v:.2f}"
            )

        if dist_m <= -3.0 or control_dist_m <= -5.0:
            passed_tag = "tl_passed_ignore_after_crossing" if control_dist_m < 0.0 else "tl_passed_ignore"
            self.mark_stopline_passed(line_id)
            self.update_stopline_debug(
                line_id,
                {
                    "line_dist": dist_m,
                    "control_dist": control_dist_m,
                    "stop_before": stop_before_m,
                    "lat": lat_m,
                },
                f"{reason}|{passed_tag}",
            )
            self.last_tl_stopline_debug["passed_ignore"] = True
            return target_speed, (
                f"{reason}|{passed_tag}:"
                f"id={line_id},line_dist={dist_m:.2f},control_dist={control_dist_m:.2f},lat={lat_m:.2f}"
            )

        stop_margin = 0.15
        crawl_m_base = float(line.get("crawl_m", 14.0))
        crawl_speed = float(line.get("crawl_speed_mps", 1.10))
        approach_speed = float(line.get("approach_speed_mps", 4.00))

        try:
            base_speed = float(target_speed or 0.0)
        except Exception:
            base_speed = 0.0

        comfortable_decel = 2.35
        reaction_s = 0.75
        brake_distance = (v * v) / max(0.1, 2.0 * comfortable_decel)
        brake_distance += reaction_s * v
        brake_distance += 4.0

        try:
            pitch = float(self.ego.get_transform().rotation.pitch)
            if abs(pitch) >= 2.5:
                brake_distance += 4.0
        except Exception:
            pass

        dynamic_crawl_m = max(crawl_m_base, brake_distance)

        if yellow_control:
            safe_stop_m = brake_distance + 1.5
            too_close_to_stop_m = max(3.0, 0.85 * v)
            if control_dist_m > 20.0:
                cap = 5.0 if control_dist_m <= 60.0 else 6.0
                return cap, (
                    f"{reason}|yellow_tl_stopline_far_prepare:"
                    f"id={line_id},line_dist={dist_m:.2f},control_dist={control_dist_m:.2f},"
                    f"stop_before={stop_before_m:.2f},lat={lat_m:.2f},cap={cap:.2f},v={v:.2f}"
                )
            if 0.0 < control_dist_m < too_close_to_stop_m and v > 1.8:
                cap = min(max(float(target_speed or 0.0), 1.2), 3.0)
                return cap, (
                    f"{reason}|yellow_tl_stopline_too_close_continue:"
                    f"id={line_id},line_dist={dist_m:.2f},control_dist={control_dist_m:.2f},"
                    f"stop_before={stop_before_m:.2f},lat={lat_m:.2f},cap={cap:.2f},v={v:.2f}"
                )
            if control_dist_m > safe_stop_m:
                cap = min(float(target_speed or approach_speed), approach_speed)
                return cap, (
                    f"{reason}|yellow_tl_stopline_prepare:"
                    f"id={line_id},line_dist={dist_m:.2f},control_dist={control_dist_m:.2f},"
                    f"stop_before={stop_before_m:.2f},lat={lat_m:.2f},cap={cap:.2f},v={v:.2f}"
                )

        visual_red_stop_zero = (
            base_speed <= 0.05
            and (
                "red_light_no_sensor_overhead_stop" in reason_l
                or "red_light_no_sensor_stop" in reason_l
                or "red_light_visual_forced_stop" in reason_l
                or "driver_only_decision_stop:red_light" in reason_l
                or "red_light_no_sensor_crawl_visual" in reason_l
            )
        )

        def choose_speed(limit_speed):
            if visual_red_stop_zero:
                return float(limit_speed)
            if base_speed > 0.05:
                return min(base_speed, float(limit_speed))
            return float(limit_speed)

        near_line_stop_control = abs(float(lat_m)) <= 3.0 and fb_control_dist_m <= 1.0
        if green_release_text:
            if near_line_stop_control and fb_control_dist_m >= 0.0:
                return 0.0, (
                    f"{reason}|tl_green_wait_stable_near_line:"
                    f"id={line_id},line_dist={dist_m:.2f},control_dist={control_dist_m:.2f},"
                    f"stop_before={stop_before_m:.2f},lat={lat_m:.2f},v={v:.2f},{fb_reason}"
                )
            return max(float(target_speed or 0.0), 1.2), (
                f"{reason}|tl_green_wait_stable_release_red_profile:"
                f"id={line_id},line_dist={dist_m:.2f},control_dist={control_dist_m:.2f},"
                f"stop_before={stop_before_m:.2f},lat={lat_m:.2f},v={v:.2f},{fb_reason}"
            )

        if current_red_yellow and fb_control_dist_m <= 0.3:
            self.last_tl_stopline_debug["passed_ignore"] = False
            return 0.0, (
                f"{reason}|tl_red_front_bumper_final_stop:"
                f"id={line_id},line_dist={dist_m:.2f},control_dist={control_dist_m:.2f},"
                f"stop_before={stop_before_m:.2f},lat={lat_m:.2f},v={v:.2f},{fb_reason}"
            )

        if current_red_yellow and fb_control_dist_m <= 1.0:
            final_target = 0.0 if v > 0.3 else 0.1
            self.last_tl_stopline_debug["passed_ignore"] = False
            return final_target, (
                f"{reason}|tl_red_front_bumper_final_brake:"
                f"id={line_id},line_dist={dist_m:.2f},control_dist={control_dist_m:.2f},"
                f"stop_before={stop_before_m:.2f},lat={lat_m:.2f},v={v:.2f},{fb_reason}"
            )

        visual_crawl_override = (
            "red_light_no_sensor_crawl_visual" in reason_l
            and base_speed <= 1.50
        )

        if memory_profile_active and control_dist_m > 45.0:
            capped = 5.5
            tag = "tl_red_far_approach_memory"
            if "driver_only_decision_stop" in reason_l or "decision_stop" in reason_l:
                tag = "tl_red_far_approach_memory_override_stop"
            return capped, (
                f"{reason}|{tag}:"
                f"id={line_id},line_dist={dist_m:.2f},control_dist={control_dist_m:.2f},"
                f"stop_before={stop_before_m:.2f},lat={lat_m:.2f},cap={capped:.2f},v={v:.2f}"
            )

        if memory_profile_active and control_dist_m > 30.0:
            capped = 4.5
            return capped, (
                f"{reason}|tl_red_mid_approach_memory:"
                f"id={line_id},line_dist={dist_m:.2f},control_dist={control_dist_m:.2f},"
                f"stop_before={stop_before_m:.2f},lat={lat_m:.2f},cap={capped:.2f},v={v:.2f}"
            )

        if current_red_yellow and control_dist_m > 30.0:
            if control_dist_m > 60.0:
                profile_speed = 7.0
            else:
                profile_speed = 6.0

            capped = self.clamp(max(6.0, min(8.0, choose_speed(profile_speed))), 0.0, self.max_speed_mps)
            tag = "tl_red_far_approach_no_stop_recover"

            return capped, (
                f"{reason}|{tag}:"
                f"id={line_id},line_dist={dist_m:.2f},control_dist={control_dist_m:.2f},"
                f"stop_before={stop_before_m:.2f},lat={lat_m:.2f},cap={capped:.2f},"
                f"v={v:.2f},brake_dist={brake_distance:.1f}"
            )

        if memory_profile_active and control_dist_m > 10.0:
            capped = 2.5
            return capped, (
                f"{reason}|tl_red_near_approach_memory:"
                f"id={line_id},line_dist={dist_m:.2f},control_dist={control_dist_m:.2f},"
                f"stop_before={stop_before_m:.2f},lat={lat_m:.2f},cap={capped:.2f},"
                f"v={v:.2f},brake_dist={brake_distance:.1f}"
            )

        if current_red_yellow and control_dist_m > 10.0:
            capped = choose_speed(2.5)
            if visual_crawl_override:
                capped = max(capped, 2.0)

            return capped, (
                f"{reason}|tl_red_approach:"
                f"id={line_id},line_dist={dist_m:.2f},control_dist={control_dist_m:.2f},"
                f"stop_before={stop_before_m:.2f},lat={lat_m:.2f},cap={capped:.2f},"
                f"v={v:.2f},brake_dist={brake_distance:.1f}"
            )

        if memory_profile_active and control_dist_m > 2.5:
            capped = 1.0 if v > 0.5 else 0.7
            return capped, (
                f"{reason}|tl_red_creep_to_stopline_memory:"
                f"id={line_id},line_dist={dist_m:.2f},control_dist={control_dist_m:.2f},"
                f"stop_before={stop_before_m:.2f},lat={lat_m:.2f},cap={capped:.2f},"
                f"v={v:.2f},brake_dist={brake_distance:.1f}"
            )

        if current_red_yellow and control_dist_m > 3.0:
            capped = choose_speed(1.2)
            if v < 0.20:
                capped = min(max(capped, 0.8), 1.2)

            return capped, (
                f"{reason}|tl_red_creep_to_stopline:"
                f"id={line_id},line_dist={dist_m:.2f},control_dist={control_dist_m:.2f},"
                f"stop_before={stop_before_m:.2f},lat={lat_m:.2f},cap={capped:.2f},"
                f"v={v:.2f},brake_dist={brake_distance:.1f}"
            )

        if control_dist_m <= dynamic_crawl_m:
            capped = choose_speed(crawl_speed)
            if control_dist_m <= 5.0:
                if control_dist_m > 3.0 and v < 0.20:
                    capped = min(max(capped, 0.55), 0.85)
                elif control_dist_m > 1.0:
                    capped = min(max(capped, 0.35), 0.65)
                else:
                    capped = min(capped, max(0.25, 0.18 + 0.08 * control_dist_m))

            return capped, (
                f"{reason}|tl_red_creep_to_stopline:"
                f"id={line_id},line_dist={dist_m:.2f},control_dist={control_dist_m:.2f},"
                f"stop_before={stop_before_m:.2f},lat={lat_m:.2f},cap={capped:.2f},"
                f"v={v:.2f},brake_dist={brake_distance:.1f}"
            )

        capped = choose_speed(approach_speed)
        return capped, (
            f"{reason}|tl_red_approach:"
            f"id={line_id},line_dist={dist_m:.2f},control_dist={control_dist_m:.2f},"
            f"stop_before={stop_before_m:.2f},lat={lat_m:.2f},cap={capped:.2f},"
            f"v={v:.2f},brake_dist={brake_distance:.1f}"
        )


    def reset_longitudinal_memory(self):
        self.last_throttle_cmd = 0.0
        self.last_brake_cmd = 0.0

        try:
            self.last_smoothed_target_speed_mps = 0.0
            self.last_smoothed_target_time = time.time()
        except Exception:
            pass

    def apply_planner_speed_hint_cap(self, target_speed, reason):
        now = time.time()
        reason_l = str(reason or "").lower()

        passed_override = self.maybe_override_passed_tl_decision(
            target_speed,
            reason,
            use_debug_passed=False,
        )
        if passed_override is not None:
            target_speed, reason = passed_override
            reason_l = str(reason or "").lower()

        # GREEN_WAIT_NO_RED_HOLD_FIX:
        # green_wait_stable sadece kırmızıdan dolayı TL hold aktifse aracı bekletmeli.
        # Araç hareket halindeyken/hold yokken yeşil bekleme kararı full STOP'a çevrilirse
        # yokuşta ve düz yolda gereksiz 1 saniyelik dur-kalk yapıyor.
        tl_hold_active_now = (
            bool(getattr(self, "tl_stop_hold_active", False))
            and now < float(getattr(self, "tl_stop_hold_until", 0.0) or 0.0)
        )

        if (
            ("green_wait_stable" in reason_l or "decision_hold_after_stop:green_wait_stable" in reason_l)
            and not tl_hold_active_now
        ):
            self.tl_stop_hold_active = False
            self.tl_stop_hold_until = 0.0
            self.last_brake_cmd = 0.0

            # Yeşil henüz confirmed değilse durma değil, kontrollü devam.
            # Yokuşta gereksiz full brake yapmasın ama hız da sınırsız kalmasın.
            return 4.0, f"{reason}|green_wait_no_red_hold_ignore"

        # GREEN_RELEASE_POST_IGNORE_FIX:
        # Confirmed green ile kalktıktan sonra aynı kavşağın eski red/yellow'u aracı
        # tekrar kilitlemesin. Ancak önümüzde yeni/manual stopline varsa ASLA ignore etme;
        # aksi halde yakın ikinci kırmızıda geçme yapar.
        red_yellow_reason = (
            "yellow_light_detected" in reason_l
            or "red_light_no_sensor" in reason_l
            or "red_light_" in reason_l
        )
        current_red_yellow_now = self.current_tl_red_yellow(reason)

        manual_stopline_ahead = False
        if red_yellow_reason:
            try:
                manual_stopline_ahead = self.nearest_manual_tl_stopline() is not None
            except Exception:
                manual_stopline_ahead = False

        if (
            now < float(getattr(self, "tl_post_green_ignore_until", 0.0) or 0.0)
            and red_yellow_reason
            and not current_red_yellow_now
            and not manual_stopline_ahead
            and "green_light_confirmed_stable" not in reason_l
        ):
            self.tl_stop_hold_active = False
            self.tl_stop_hold_until = 0.0
            self.last_brake_cmd = 0.0
            return max(float(target_speed or 0.0), 2.0), f"{reason}|post_green_same_intersection_tl_ignore"

        try:
            current_red_yellow_for_stopline = (
                "red_light" in reason_l
                or "yellow_light" in reason_l
                or "traffic_light_red" in reason_l
                or "traffic_light_yellow" in reason_l
            )

            if self.manual_tl_red_active() or current_red_yellow_for_stopline:
                target_speed, reason = self.apply_manual_tl_stopline_cap(target_speed, reason)
                reason_s = str(reason)

                passed_override = self.maybe_override_passed_tl_decision(target_speed, reason)
                if passed_override is not None:
                    target_speed, reason = passed_override
                    reason_s = str(reason)

                if (
                    "tl_stopline_stop" in reason_s
                    or "tl_red_stop_at_line" in reason_s
                    or "tl_red_final_stop_at_line" in reason_s
                ):
                    self.tl_stop_hold_active = True
                    self.tl_stop_hold_until = now + 8.0
                    self.last_brake_cmd = 0.0
                    self.last_throttle_cmd = 0.0
                    return 0.0, reason

                if "tl_stopline_" in reason_s or "tl_red_" in reason_s or "tl_passed_ignore" in reason_s:
                    return float(target_speed), reason

                if "tl_red_no_stopline_fallback_" in reason_s:
                    return float(target_speed), reason

        except Exception as exc:
            try:
                self.get_logger().warning(f"manual stopline priority failed: {exc}")
            except Exception:
                pass

        latest_reason = ""
        try:
            latest_reason = str((getattr(self, "latest_decision", {}) or {}).get("reason", "") or "").lower()
        except Exception:
            pass

        green_confirmed = self.is_verified_green_decision(reason=f"{reason}|{latest_reason}")

        release = self.maybe_release_no_stopline_hold(target_speed, reason)
        if release is not None:
            return release

        if green_confirmed and not bool(getattr(self, "active_tl_stopline_id", None)):
            self.clear_red_hold(mark_green_release=True)
            if "tl_red_hold_release_verified_green" not in str(reason):
                reason = f"{reason}|tl_red_hold_release_verified_green"

        if (
            bool(getattr(self, "tl_stop_hold_active", False))
            and now < float(getattr(self, "tl_stop_hold_until", 0.0) or 0.0)
            and not green_confirmed
            and not bool(getattr(self, "active_tl_stopline_id", None))
        ):
            release = self.maybe_release_no_stopline_hold(target_speed, reason)
            if release is not None:
                return release
            return 0.0, f"{reason}|tl_red_hold_unknown_wait:no_active_stopline"
        target_speed, reason = self.apply_manual_tl_stopline_cap(target_speed, reason)
        if (
            "tl_stopline_stop" in str(reason)
            or "tl_red_stop_at_line" in str(reason)
            or "tl_red_final_stop_at_line" in str(reason)
        ):
            return target_speed, reason

        green_release_window = (
            bool(getattr(self, "tl_green_release_active", False))
            and bool(getattr(self, "tl_green_release_from_hold", False))
            and now < float(getattr(self, "green_release_force_until", 0.0) or 0.0)
            and (
                self.latest_tl_state() == "green"
                or "green_light_confirmed_stable" in reason_l
                or "tl_green_release_no_stopline_hold" in reason_l
                or "tl_no_stopline_hold_go_release" in reason_l
            )
        )

        if green_release_window:
            started = float(getattr(self, "green_release_started_at", 0.0) or 0.0)
            age = max(0.0, now - started) if started > 0.0 else 0.0
            try:
                speed_now = float(self.get_speed())
            except Exception:
                speed_now = 0.0
            if speed_now > 2.0:
                self.tl_green_release_active = False
                self.tl_green_release_from_hold = False
                return target_speed, f"{reason}|tl_green_smooth_release_skip_moving_vehicle"
            if age > 1.0 or not bool(getattr(self, "red_hold_active", False)):
                self.tl_green_release_active = False
                self.tl_green_release_from_hold = False
                return max(float(target_speed or 0.0), 4.0), f"{reason}|tl_green_smooth_release_fast_recover"
            release_cap = 4.0 + min(1.5, age * 2.0)
            try:
                target_speed_f = float(target_speed or 0.0)
            except Exception:
                target_speed_f = 0.0
            if target_speed_f > release_cap:
                return release_cap, f"{reason}|tl_green_smooth_release:age={age:.1f},cap={release_cap:.2f}"
        elif now >= float(getattr(self, "green_release_force_until", 0.0) or 0.0):
            self.tl_green_release_active = False
            self.tl_green_release_from_hold = False

        """
        Planner speed_hint eskiden sadece resolve_target_speed sonundaki normal dönüşte
        uygulanıyordu. Trafik ışığı guard içindeki erken return'ler bunu bypass ediyordu.
        Bu yüzden park yaklaşımında 0.75 m/s hedefinden bir anda 5.5 m/s hedefe zıplıyordu.

        Bu fonksiyon tick seviyesinde tekrar cap uygular; hiçbir erken return planner hızını aşamaz.
        """
        if not self.planner_speed_hint_enabled:
            return target_speed, reason

        planner_target = self.get_fresh_planner_target()
        if planner_target is None:
            return target_speed, reason

        speed_hint = planner_target.get("speed_hint_mps")

        if planner_target.get("speed_hint_kmh") is not None:
            try:
                speed_hint = self.kmh_to_mps(planner_target.get("speed_hint_kmh"))
            except Exception:
                pass

        if speed_hint is None:
            return target_speed, reason

        try:
            speed_hint = float(speed_hint)
            target_speed = float(target_speed)
        except Exception:
            return target_speed, reason

        # TL_GREEN_START_FIX:
        # Yeşil kalkış anında planner speed_hint 0/düşük kalırsa decision GO ezilmesin.
        reason_l = str(reason or "").lower()
        green_release_active = (
            bool(getattr(self, "tl_green_release_active", False))
            and bool(getattr(self, "tl_green_release_from_hold", False))
            and time.time() < float(getattr(self, "green_release_force_until", 0.0) or 0.0)
            and (
                self.latest_tl_state() == "green"
                or "green_light_confirmed_stable" in reason_l
                or "tl_green_release_no_stopline_hold" in reason_l
                or "tl_no_stopline_hold_go_release" in reason_l
            )
        )

        if green_release_active and target_speed > 0.30:
            try:
                current_speed = float(self.get_speed())
            except Exception:
                current_speed = 0.0

            min_green_start = min(
                float(target_speed),
                max(0.8, float(getattr(self, "green_release_min_start_speed_mps", 3.0))),
            )

            if current_speed < 0.45 and speed_hint < min_green_start:
                reason = (
                    f"{reason}|planner_speed_hint_ignored_for_green_release:"
                    f"{speed_hint * 3.6:.1f}kmh->{min_green_start * 3.6:.1f}kmh"
                )
                return min_green_start, reason

        capped = min(target_speed, speed_hint)

        if capped < target_speed - 0.01 and "planner_speed_hint" not in str(reason):
            reason = f"{reason}|planner_speed_hint:{speed_hint * 3.6:.1f}kmh/{speed_hint:.2f}mps"

        return capped, reason

    def apply_target_speed_smoothing(self, target_speed, reason):
        target_speed = float(target_speed)
        reason_s = str(reason or "")
        reason_l = reason_s.lower()

        tl_stopline_hard_cap_active = (
            "tl_stopline_" in reason_l
            or "tl_red_" in reason_l
            or "yellow_tl_stopline_" in reason_l
        )

        if tl_stopline_hard_cap_active:
            self.last_smoothed_target_speed_mps = target_speed
            self.last_smoothed_target_time = time.time()
            return target_speed, f"{reason}|target_smooth_bypass_for_tl_stopline:{target_speed:.2f}"

        if not self.target_speed_smoothing_enabled:
            self.last_smoothed_target_speed_mps = target_speed
            self.last_smoothed_target_time = time.time()
            return target_speed, reason

        now = time.time()

        if target_speed <= 0.01:
            self.last_smoothed_target_speed_mps = 0.0
            self.last_smoothed_target_time = now
            return 0.0, reason

        if self.last_smoothed_target_speed_mps is None:
            self.last_smoothed_target_speed_mps = target_speed
            self.last_smoothed_target_time = now
            return target_speed, reason

        dt = now - float(self.last_smoothed_target_time or now)
        dt = self.clamp(dt, 0.02, 0.20)
        prev = float(self.last_smoothed_target_speed_mps)

        if (
            ("green_light_detected" in reason_l or "green_light_confirmed_stable" in reason_l)
            and prev <= 0.05
            and target_speed > 0.10
        ):
            kickoff = min(
                target_speed,
                max(float(self.green_release_min_start_speed_mps), target_speed * 0.35),
            )
            self.last_smoothed_target_speed_mps = kickoff
            self.last_smoothed_target_time = now
            reason = f"{reason}|green_release_smoothing_bypass:{kickoff:.2f}"
            return kickoff, reason

        if target_speed > prev:
            max_step = max(0.05, float(self.target_speed_accel_limit_mps2) * dt)
            smoothed = min(target_speed, prev + max_step)
        else:
            max_step = max(0.08, float(self.target_speed_decel_limit_mps2) * dt)
            smoothed = max(target_speed, prev - max_step)

        self.last_smoothed_target_speed_mps = smoothed
        self.last_smoothed_target_time = now

        if abs(smoothed - target_speed) > 0.03:
            reason = f"{reason}|target_smooth:{target_speed:.2f}->{smoothed:.2f}"

        return smoothed, reason


    def get_stage(self):
        if not self.latest_mission:
            return None
        return self.latest_mission.get("stage")

    def get_fresh_planner_target(self):
        if not self.use_planner_local_target:
            return None

        if self.latest_local_target is None:
            return None

        if time.time() - self.last_local_target_time > self.planner_fresh_timeout_s:
            return None

        return self.latest_local_target

    def get_mission_objective_target(self):
        if not self.latest_mission:
            return None

        target = self.latest_mission.get("objective_target")
        if isinstance(target, dict):
            return target

        target = self.latest_mission.get("target")
        if isinstance(target, dict):
            return target

        return None

    def get_target(self):
        if self.use_planner_local_target and self.planner_local_target_as_destination:
            planner_target = self.get_fresh_planner_target()
            if planner_target is not None:
                return planner_target

        return self.get_mission_objective_target()

    def get_target_key(self):
        target = self.get_target()
        if not target or not self.latest_mission:
            return None

        planner_key = target.get("_planner_key")
        if planner_key and self.planner_local_target_as_destination:
            return str(planner_key)

        objective_index = self.latest_mission.get(
            "objective_index",
            self.latest_mission.get("route_index", self.latest_mission.get("task_index")),
        )

        objective_kind = self.latest_mission.get(
            "objective_kind",
            self.latest_mission.get("route_kind", ""),
        )

        try:
            carla_x = round(float(target.get("carla_x", target.get("lon", 0.0))), 3)
            carla_y = round(float(target.get("carla_y", target.get("lat", 0.0))), 3)
        except Exception:
            carla_x = target.get("carla_x", target.get("lon", 0.0))
            carla_y = target.get("carla_y", target.get("lat", 0.0))

        return (
            str(self.latest_mission.get("stage")) + "|" +
            str(objective_index) + "|" +
            str(objective_kind) + "|" +
            str(target.get("name")) + "|" +
            str(carla_x) + "|" +
            str(carla_y)
        )

    def mission_geo_to_carla_location_near_ego(self, target):
        """
        Town03 simülasyonunda mission dosyamız CARLA local x/y kullanıyor.
        Eski kod target lat/lon bilgisini CARLA geolocation sanıp dönüşüm yapıyordu.
        Bu da 10 milyon metre gibi saçma mesafelere ve yanlış BasicAgent hedefine yol açıyordu.

        Eğer target içinde carla_x/carla_y varsa direkt local CARLA Location döndür.
        Yoksa legacy gerçek GPS davranışına geri düş.
        """
        try:
            if target.get("carla_x") is not None and target.get("carla_y") is not None:
                x = float(target.get("carla_x"))
                y = float(target.get("carla_y"))
                z = target.get("carla_z", None)

                if z is None:
                    ego_z = self.ego.get_location().z
                    z = ego_z
                else:
                    z = float(z)

                return self.carla.Location(x=x, y=y, z=z + 0.2)
        except Exception as exc:
            self.get_logger().warning(f"carla_x/carla_y target parse hatası: {exc}")

        ego_loc = self.ego.get_location()

        base_geo = self.map.transform_to_geolocation(ego_loc)

        geo_x = self.map.transform_to_geolocation(
            self.carla.Location(x=ego_loc.x + 1.0, y=ego_loc.y, z=ego_loc.z)
        )
        geo_y = self.map.transform_to_geolocation(
            self.carla.Location(x=ego_loc.x, y=ego_loc.y + 1.0, z=ego_loc.z)
        )

        lat0 = float(base_geo.latitude)
        lon0 = float(base_geo.longitude)

        lat_dx = float(geo_x.latitude) - lat0
        lon_dx = float(geo_x.longitude) - lon0
        lat_dy = float(geo_y.latitude) - lat0
        lon_dy = float(geo_y.longitude) - lon0

        target_lat = float(target["lat"])
        target_lon = float(target["lon"])

        dlat = target_lat - lat0
        dlon = target_lon - lon0

        det = lat_dx * lon_dy - lat_dy * lon_dx

        if abs(det) < 1e-16:
            self.get_logger().warning("Geo inverse det çok küçük.")
            return ego_loc

        dx = (dlat * lon_dy - lat_dy * dlon) / det
        dy = (lat_dx * dlon - dlat * lon_dx) / det

        dx = self.clamp(dx, -500.0, 500.0)
        dy = self.clamp(dy, -500.0, 500.0)

        return self.carla.Location(x=ego_loc.x + dx, y=ego_loc.y + dy, z=ego_loc.z)


    def get_turn_direction_to_target(self, target):
        """
        Ego'dan hedefe göre kaba dönüş niyeti:
        - left  : hedef ego'nun solunda kalıyor
        - right : hedef ego'nun sağında kalıyor
        - straight: büyük yan sapma yok

        Bu sadece şerit/yaklaşım seçimi için kullanılır; asıl rotayı BasicAgent üretir.
        """
        try:
            raw_loc = self.mission_geo_to_carla_location_near_ego(target)
            ego_tf = self.ego.get_transform()
            ego_loc = ego_tf.location
            fwd = ego_tf.get_forward_vector()
            right = ego_tf.get_right_vector()

            vx = raw_loc.x - ego_loc.x
            vy = raw_loc.y - ego_loc.y

            forward_dot = vx * fwd.x + vy * fwd.y
            right_dot = vx * right.x + vy * right.y

            # Sağ pozitif, sol negatif.
            angle_deg = math.degrees(math.atan2(right_dot, max(0.001, forward_dot)))

            if angle_deg < -22.0:
                return "left", angle_deg
            if angle_deg > 22.0:
                return "right", angle_deg

            return "straight", angle_deg

        except Exception as exc:
            self.get_logger().warning(f"turn direction hesaplanamadı: {exc}", throttle_duration_sec=1.0)
            return "unknown", 0.0

    def get_same_direction_adjacent_lane(self, wp, turn_direction):
        """
        Mümkünse aynı yöndeki komşu şeridi seç.
        Sol dönüşte sol şerit, sağ dönüşte sağ şerit tercih edilir.
        """
        try:
            if wp is None:
                return None

            if turn_direction == "left":
                cand = wp.get_left_lane()
            elif turn_direction == "right":
                cand = wp.get_right_lane()
            else:
                return None

            if cand is None:
                return None

            if cand.lane_type != self.carla.LaneType.Driving:
                return None

            # CARLA'da aynı yöndeki lane'ler genelde aynı lane_id işaretindedir.
            try:
                if int(cand.lane_id) * int(wp.lane_id) <= 0:
                    return None
            except Exception:
                pass

            return cand

        except Exception:
            return None

    def shifted_location_from_waypoint(self, wp, lateral_shift_m):
        loc = wp.transform.location

        if abs(float(lateral_shift_m)) < 0.05:
            return self.carla.Location(x=loc.x, y=loc.y, z=loc.z + 0.2)

        right_vec = wp.transform.get_right_vector()
        shifted_x = loc.x + right_vec.x * float(lateral_shift_m)
        shifted_y = loc.y + right_vec.y * float(lateral_shift_m)

        # Shift sonrası tekrar yola projekte et ki off-road hedef verilmesin.
        shifted = self.carla.Location(x=shifted_x, y=shifted_y, z=loc.z + 0.2)

        try:
            shifted_wp = self.map.get_waypoint(
                shifted,
                project_to_road=True,
                lane_type=self.carla.LaneType.Driving,
            )

            if shifted_wp is not None:
                sloc = shifted_wp.transform.location
                return self.carla.Location(x=sloc.x, y=sloc.y, z=sloc.z + 0.2)

        except Exception:
            pass

        return shifted

    def destination_from_target(self, target):
        """
        SAFE_LANE_CENTER_FIX:
        Town03'te target adında 'sag/right/park' geçince yapılan otomatik sağ şerit/sağ shift
        aracı kaldırım, tabela, direk ve bina tarafına fazla yaklaştırıyordu.

        Bu yüzden hedefi artık doğrudan CARLA'nın sürüş şeridi merkezine projekte ediyoruz.
        Park hedefinde bile ekstra sağa shift yok; park noktası mission dosyasında zaten belirleniyor.
        """
        raw_loc = self.mission_geo_to_carla_location_near_ego(target)

        wp = self.map.get_waypoint(
            raw_loc,
            project_to_road=True,
            lane_type=self.carla.LaneType.Driving,
        )

        stage = str(self.get_stage() or "")
        target_name = str(target.get("name", "")).lower()

        if wp is None:
            self.get_logger().warning("Target waypoint bulunamadı, raw location kullanılacak.")
            return raw_loc

        loc = wp.transform.location
        dest = self.carla.Location(x=loc.x, y=loc.y, z=loc.z + 0.2)

        self.get_logger().info(
            f"Destination lane approach SAFE_CENTER: target={target.get('name')} "
            f"stage={stage} shift=0.00 reason=no_auto_right_shift "
            f"dest=({dest.x:.2f},{dest.y:.2f},{dest.z:.2f})",
            throttle_duration_sec=0.5,
        )

        return dest

    def is_planner_local_target(self, target):
        if not isinstance(target, dict):
            return False

        if target.get("_planner_key"):
            return True

        if str(target.get("kind", "")).lower() == "planner_local":
            return True

        if str(target.get("name", "")).startswith("planner_local_"):
            return True

        return False

    def distance_to_location(self, loc):
        try:
            ego_loc = self.ego.get_location()
            return math.hypot(float(ego_loc.x) - float(loc.x), float(ego_loc.y) - float(loc.y))
        except Exception:
            return None

    def should_hold_current_destination(self, target, key, new_dest):
        """
        Planner local target 10 Hz akıyor. Eski davranışta her birkaç metrede
        BasicAgent.set_destination tekrar çalışıyordu. Bu, BasicAgent'ın iç route
        planını sürekli sıfırladığı için araçta tekleme/silkelenme yapıyordu.

        Yeni davranış:
          - route_id değişirse hemen güncelle.
          - aktif destination'a yeterince yaklaştıysa güncelle.
          - yeni destination çok ilerideyse ve minimum süre geçtiyse güncelle.
          - aksi halde mevcut BasicAgent hedefini tut.
        """
        if not self.planner_destination_hold_enabled:
            return False

        if not self.is_planner_local_target(target):
            return False

        if self.active_target_key is None or self.active_destination is None:
            return False

        now = time.time()

        try:
            route_id = target.get("route_id")
        except Exception:
            route_id = None

        if route_id is not None and self.active_planner_route_id is not None:
            if route_id != self.active_planner_route_id:
                return False

        active_dist = self.distance_to_location(self.active_destination)
        if active_dist is None:
            return False

        if active_dist <= self.planner_destination_reached_m:
            return False

        try:
            new_delta = math.hypot(
                float(new_dest.x) - float(self.active_destination.x),
                float(new_dest.y) - float(self.active_destination.y),
            )
        except Exception:
            new_delta = 999.0

        elapsed = now - float(self.last_agent_destination_set_time or 0.0)

        if (
            elapsed >= self.planner_destination_update_min_interval_s
            and new_delta >= self.planner_destination_update_min_distance_m
        ):
            return False

        self.route_status = (
            f"planner_destination_hold:"
            f"active_dist={active_dist:.1f},new_delta={new_delta:.1f},elapsed={elapsed:.1f}"
        )
        return True

    def set_agent_destination_if_needed(self):
        target = self.get_target()
        key = self.get_target_key()

        if target is None or key is None:
            self.route_status = "mission_target_missing"
            return False

        dest = self.destination_from_target(target)

        if key == self.active_target_key:
            return True

        if self.should_hold_current_destination(target, key, dest):
            return True

        start = self.ego.get_location()

        try:
            self.agent.set_destination(dest)
        except TypeError:
            self.agent.set_destination(dest, start_location=start)
        except Exception:
            try:
                self.agent.set_destination(start, dest)
            except Exception as e:
                self.route_status = f"set_destination_failed:{e}"
                self.get_logger().error(self.route_status)
                return False

        self.active_target_key = key
        self.active_destination = dest
        self.last_agent_destination_set_time = time.time()

        if self.is_planner_local_target(target):
            self.active_planner_route_id = target.get("route_id")

        self.route_status = f"basic_agent_route_to:{target.get('name')}"

        self.get_logger().info(
            f"BasicAgent destination set: stage={self.get_stage()} "
            f"target={target.get('name')} dest=({dest.x:.2f},{dest.y:.2f},{dest.z:.2f})"
        )

        return True

    def smooth_stop_control(self, reason=""):
        """
        STOP / hard-stop durumları için güvenli kontrol üretir.

        Bu metod daha önce tick() içinde çağrılıyordu ama dosyada yoktu.
        Eksik olduğu için route_agent crash oluyor ve araç komut alamıyordu.
        """
        try:
            import carla
            control = carla.VehicleControl()
        except Exception:
            # CARLA import beklenmedik şekilde yoksa yine de crash etme.
            class _Control:
                pass
            control = _Control()

        text = str(reason or "").lower()
        if (
            bool(getattr(self, "tl_waiting_for_verified_green", False))
            or "tl_wait_green_latch_" in text
        ) and "tl_wait_green_latch_release_verified_green" not in text:
            control.throttle = 0.0
            control.brake = 1.0
            control.hand_brake = True
            control.reverse = False
            control.manual_gear_shift = False
            self.last_throttle_cmd = 0.0
            self.last_brake_cmd = 1.0
            self.last_steer_cmd = float(control.steer)
            return control

        red_stop = "red_light_" in text and (
            "stop" in text
            or "visual_stop" in text
            or "no_distance_bottom" in text
        )

        prev_brake = float(getattr(self, "last_brake_cmd", 0.0) or 0.0)
        prev_steer = float(getattr(self, "last_steer_cmd", 0.0) or 0.0)

        control.throttle = 0.0
        control.steer = self.clamp(prev_steer, -0.35, 0.35)

        soft_tl_hold = (
            "tl_red_hold_unknown_wait" in text
            or "tl_red_hold_keep_on_unknown" in text
            or "tl_red_hold_until_verified_green" in text
            or "tl_red_no_stopline_fallback_stop" in text
        )

        strong_tl_line_stop = (
            "tl_red_final_brake_to_line" in text
            or "tl_red_final_stop_at_line" in text
            or "tl_red_front_bumper_final_brake" in text
            or "tl_red_front_bumper_final_stop" in text
            or "tl_red_overshoot_recover_stop" in text
            or "tl_red_overshoot_recover_hold" in text
            or "tl_red_hold_stopped_after_overshoot" in text
            or "tl_red_overshoot_hold_until_green" in text
            or "tl_crossed_release_suppressed_red_active" in text
        )

        def _reason_metric(key, default=None):
            try:
                marker = key + "="
                i = text.find(marker)
                if i < 0:
                    return default
                j = i + len(marker)
                k = j
                while k < len(text) and text[k] not in ",| ":
                    k += 1
                return float(text[j:k])
            except Exception:
                return default

        if strong_tl_line_stop:
            try:
                speed_now = float(self.get_speed())
            except Exception:
                speed_now = 0.0
            control_dist = _reason_metric("control_dist")
            min_brake = 0.30
            if "tl_red_overshoot_recover_" in text:
                min_brake = 0.45
            if "tl_red_hold_stopped_after_overshoot" in text or "tl_red_overshoot_hold_until_green" in text:
                min_brake = max(min_brake, 0.45)
            if control_dist is not None and control_dist <= 1.0 and speed_now > 0.5:
                min_brake = max(min_brake, 0.45)
            if control_dist is not None and control_dist <= 0.5:
                min_brake = max(min_brake, 0.65)
            if control_dist is not None and control_dist <= 1.5 and speed_now > 0.7:
                min_brake = max(min_brake, 0.55)
            if control_dist is not None and control_dist <= 0.7 and speed_now > 0.3:
                min_brake = max(min_brake, 0.75)
            control.brake = self.clamp(max(prev_brake + 0.18, min_brake), min_brake, 0.85)
        elif soft_tl_hold:
            try:
                speed_now = float(self.get_speed())
            except Exception:
                speed_now = 0.0
            max_brake = 0.45 if speed_now <= 0.25 else 0.65
            control.brake = self.clamp(prev_brake + 0.08, 0.25, max_brake)
        elif red_stop or "tl_stopline_stop" in text:
            control.brake = self.clamp(prev_brake + 0.10, 0.22, 0.65)
        elif "tl_red_stop_at_line" in text or "tl_red_final_stop_at_line" in text:
            control.brake = self.clamp(prev_brake + 0.10, 0.25, 0.65)
        else:
            control.brake = self.clamp(prev_brake + 0.18, 0.25, 1.0)

        control.hand_brake = False
        control.manual_gear_shift = False

        self.last_throttle_cmd = 0.0
        self.last_brake_cmd = float(control.brake)
        self.last_steer_cmd = float(control.steer)

        return control


    def is_red_light_stop_reason(self, reason):
        text = str(reason or "").lower()
        return "red_light_" in text and (
            "stop" in text
            or "visual_stop" in text
            or "no_distance_bottom" in text
        )

    def resolve_target_speed(self):
        now = time.time()

        if self.latest_mission is None or now - self.last_mission_time > 3.0:
            return 0.0, "mission_missing_or_timeout"

        stage = str(self.latest_mission.get("stage", "UNKNOWN"))
        must_stop = bool(self.latest_mission.get("must_stop", False))

        if stage in {"COMPLETED", "FAILED"}:
            return 0.0, f"mission_{stage.lower()}"

        if self.mission_stop_override and must_stop:
            return 0.0, f"mission_stop_stage:{stage}"

        if now - self.last_decision_time > 2.0:
            return 0.0, "decision_timeout"

        decision = str(self.latest_decision.get("decision", "STOP")).upper()
        raw_reason = str(self.latest_decision.get("reason", "unknown"))

        decision_speed = None
        speed_reason = "decision_speed_missing"

        try:
            if self.latest_decision.get("target_speed_kmh") is not None:
                decision_speed = self.kmh_to_mps(self.latest_decision.get("target_speed_kmh"))
                speed_reason = f"decision_speed_kmh:{float(self.latest_decision.get('target_speed_kmh')):.1f}"
            elif self.latest_decision.get("target_speed") is not None:
                decision_speed = float(self.latest_decision.get("target_speed"))
                speed_reason = f"decision_speed_mps:{decision_speed:.2f}"
        except Exception:
            decision_speed = None
            speed_reason = "decision_speed_parse_error"

        if decision == "STOP":
            return 0.0, f"driver_only_decision_stop:{raw_reason}"

        if decision_speed is None:
            decision_speed = self.slow_speed_mps if decision == "SLOW" else self.go_speed_mps

        if decision == "SLOW":
            raw_reason_l = str(raw_reason or "").lower()

            # red_light_sensor_far_slow decision tarafında 20 km/h yaklaşma demek.
            # Eski kod bütün SLOW kararlarını slow_speed_mps=10 km/h ile kırpıyordu.
            if (
                "red_light_sensor_far_slow" in raw_reason_l
                or "red_light_no_sensor_far_slow" in raw_reason_l
            ):
                target_speed = float(decision_speed)
            else:
                target_speed = min(float(decision_speed), float(self.slow_speed_mps))

            return self.clamp(target_speed, 0.0, self.max_speed_mps), (
                f"driver_only_decision_slow:{raw_reason}|{speed_reason}"
            )

        if stage == "PARKING":
            decision_speed = min(float(decision_speed), float(self.parking_speed_mps))

        target_speed = self.clamp(float(decision_speed), 0.0, self.max_speed_mps)
        return target_speed, f"driver_only_decision_go:{raw_reason}|{speed_reason}"

    def tick(self):
        target_speed, reason = self.resolve_target_speed()

        target_speed, reason = self.apply_planner_speed_hint_cap(target_speed, reason)
        if bool(getattr(self, "tl_waiting_for_verified_green", False)):
            if self.tl_wait_green_release_allowed(reason):
                release_tag = self.tl_wait_green_release_reason_tag(reason)
                self.release_wait_green_latch()
                self.clear_red_hold(mark_green_release=True)
                target_speed = max(float(target_speed or 0.0), 4.0)
                reason = f"{reason}|tl_wait_green_latch_release_verified_green|{release_tag}"
            else:
                target_speed = 0.0
                reason = self.wait_green_latch_reason(reason, getattr(self, "last_tl_stopline_debug", {}) or {})

        red_light_control = (
            "red_light_" in str(reason).lower()
            or "tl_stopline_" in str(reason).lower()
            or "tl_red_" in str(reason).lower()
            or "yellow_tl_stopline_" in str(reason).lower()
            or "tl_red_no_stopline_fallback_" in str(reason).lower()
            or "tl_red_hold_unknown_wait" in str(reason).lower()
        )
        # Kırmızı ışıkta hedef hız düşüşünü geciktirme.
        # Aksi halde 20 km/h -> 10 km/h -> 0 geçişi çok geç oluyor.
        if not red_light_control:
            target_speed, reason = self.apply_target_speed_smoothing(target_speed, reason)

        current_speed = self.get_speed()

        if time.time() < self.collision_until:
            control = self.hard_stop_control()
            reason = "collision_halt"
            target_speed = 0.0
        elif target_speed <= 0.01:
            control = self.smooth_stop_control(reason)
        else:
            ok = self.set_agent_destination_if_needed()
            if not ok:
                control = self.hard_stop_control()
                reason = "route_missing_stop"
                target_speed = 0.0
            else:
                try:
                    self.agent.set_target_speed(target_speed * 3.6)
                except Exception:
                    pass

                try:
                    control = self.agent.run_step(debug=False)
                except TypeError:
                    control = self.agent.run_step()

                basic_steer = self.clamp(control.steer, -self.max_steer, self.max_steer)
                control.steer = self.apply_lane_assist_to_steer(basic_steer, target_speed)


                # SMOOTH_LONGITUDINAL_FIX:
                # BasicAgent steer iyi ama düşük hızda gaz/fren zıplatıyor.
                # Bu yüzden direksiyon BasicAgent'ten, throttle/brake yumuşak hız kontrolünden geliyor.
                if not hasattr(self, "last_throttle_cmd"):
                    self.last_throttle_cmd = 0.0
                if not hasattr(self, "last_brake_cmd"):
                    self.last_brake_cmd = 0.0

                speed_error = float(target_speed) - float(current_speed)
                overspeed = float(current_speed) - float(target_speed)

                desired_throttle = 0.0
                desired_brake = 0.0

                reason_l = str(reason or "").lower()
                red_light_control = (
                    "red_light_" in reason_l
                    or "tl_stopline_" in reason_l
                    or "tl_red_" in reason_l
                    or "yellow_tl_stopline_" in reason_l
                    or "tl_red_no_stopline_fallback_" in reason_l
                    or "tl_red_hold_unknown_wait" in reason_l
                )
                red_light_stop_control = (
                    self.is_red_light_stop_reason(reason)
                    or "tl_stopline_stop" in reason_l
                    or "tl_red_stop_at_line" in reason_l
                    or "tl_red_final_stop_at_line" in reason_l
                    or "tl_red_final_brake_to_line" in reason_l
                    or "tl_red_front_bumper_final_stop" in reason_l
                    or "tl_red_front_bumper_final_brake" in reason_l
                    or "tl_red_overshoot_recover_stop" in reason_l
                    or "tl_red_overshoot_recover_hold" in reason_l
                    or "tl_red_no_stopline_fallback_stop" in reason_l
                    or "tl_red_hold_unknown_wait" in reason_l
                )
                red_light_approach_control = red_light_control and not red_light_stop_control

                far_red_no_stop_recover = (
                    "tl_red_far_approach_no_stop_recover" in reason_l
                    or "tl_red_far_approach_memory" in reason_l
                    or "tl_red_far_approach_memory_override_stop" in reason_l
                )

                if far_red_no_stop_recover:
                    desired_brake = 0.0
                    if current_speed < 0.30:
                        desired_throttle = 0.24
                    elif speed_error > 0.20:
                        desired_throttle = self.clamp(0.14 + 0.20 * speed_error, 0.10, 0.40)
                    else:
                        desired_throttle = 0.0

                elif red_light_stop_control:
                    desired_throttle = 0.0
                    desired_brake = self.clamp(0.12 + 0.18 * max(0.0, overspeed), 0.10, 0.75)

                elif speed_error > 0.20:
                    if red_light_approach_control:
                        if "tl_red_creep_to_stopline" in reason_l:
                            desired_throttle = 0.16 + 0.12 * min(speed_error, 1.0)
                            desired_throttle = self.clamp(desired_throttle, 0.14, 0.26)
                            desired_brake = 0.0
                        elif "crawl" in reason_l:
                            desired_throttle = 0.10 + 0.20 * speed_error
                            desired_throttle = self.clamp(desired_throttle, 0.08, 0.28)
                        elif "sensor_slow" in reason_l or "no_sensor_slow" in reason_l:
                            desired_throttle = 0.14 + 0.26 * speed_error
                            desired_throttle = self.clamp(desired_throttle, 0.10, 0.42)
                        else:
                            desired_throttle = 0.18 + 0.32 * speed_error
                            desired_throttle = self.clamp(desired_throttle, 0.12, 0.55)
                    else:
                        desired_throttle = 0.22 + 0.42 * speed_error
                        desired_throttle = self.clamp(desired_throttle, 0.20, 0.90)

                    if "tl_red_creep_to_stopline" not in reason_l:
                        desired_brake = 0.0

                elif red_light_control and overspeed > 0.10:
                    desired_throttle = 0.0
                    desired_brake = self.clamp(0.08 + 0.16 * overspeed, 0.08, 0.45)

                elif overspeed <= 0.95:
                    if current_speed < target_speed:
                        desired_throttle = 0.030 if red_light_approach_control else 0.018
                    else:
                        desired_throttle = 0.0
                    desired_brake = 0.0

                else:
                    desired_throttle = 0.0
                    if red_light_control:
                        desired_brake = self.clamp(0.10 + 0.10 * (overspeed - 0.20), 0.08, 0.45)
                    else:
                        desired_brake = self.clamp(0.05 * (overspeed - 0.95), 0.0, 0.035)

                def _slew(cur, dst, step):
                    cur = float(cur)
                    dst = float(dst)
                    step = abs(float(step))
                    if dst > cur:
                        return min(dst, cur + step)
                    if dst < cur:
                        return max(dst, cur - step)
                    return cur

                throttle_cmd = _slew(self.last_throttle_cmd, desired_throttle, 0.160)
                red_light_control = (
                    "red_light_" in str(reason).lower()
                    or "tl_stopline_" in str(reason).lower()
                    or "tl_red_" in str(reason).lower()
                    or "yellow_tl_stopline_" in str(reason).lower()
                    or "tl_red_no_stopline_fallback_" in str(reason).lower()
                    or "tl_red_hold_unknown_wait" in str(reason).lower()
                )
                brake_cmd = _slew(
                    self.last_brake_cmd,
                    desired_brake,
                    0.060 if red_light_control else 0.018,
                )

                if far_red_no_stop_recover:
                    brake_cmd = 0.0

                if "tl_red_creep_to_stopline" in str(reason).lower() and float(target_speed) > 0.20:
                    brake_cmd = 0.0

                if brake_cmd > 0.001:
                    throttle_cmd = 0.0

                self.last_throttle_cmd = throttle_cmd
                self.last_brake_cmd = brake_cmd

                control.throttle = self.clamp(throttle_cmd, 0.0, 0.90)
                red_light_control = (
                    "red_light_" in str(reason).lower()
                    or "tl_stopline_" in str(reason).lower()
                    or "tl_red_" in str(reason).lower()
                    or "yellow_tl_stopline_" in str(reason).lower()
                    or "tl_red_no_stopline_fallback_" in str(reason).lower()
                    or "tl_red_hold_unknown_wait" in str(reason).lower()
                )
                strong_line_brake = (
                    "tl_red_final_brake_to_line" in str(reason).lower()
                    or "tl_red_final_stop_at_line" in str(reason).lower()
                    or "tl_red_front_bumper_final_brake" in str(reason).lower()
                    or "tl_red_front_bumper_final_stop" in str(reason).lower()
                    or "tl_red_overshoot_recover_stop" in str(reason).lower()
                    or "tl_red_overshoot_recover_hold" in str(reason).lower()
                )
                control.brake = self.clamp(
                    brake_cmd,
                    0.0,
                    0.75 if strong_line_brake else (0.45 if red_light_control else 0.035),
                )
                control.hand_brake = False
                control.manual_gear_shift = False

        # HARD SAFETY CLAMP: araç kontrolden çıkmasın diye hız/gaz sınırı
        try:
            control.throttle = self.clamp(control.throttle, 0.0, 0.90)
            control.steer = self.clamp(control.steer, -self.max_steer, self.max_steer)
        except Exception:
            pass

        # GO kararı var ama araç duruyorsa BasicAgent bazen ilk kalkışta throttle=0/brake>0 bırakıyor.
        # Kırmızı/STOP yoksa freni temizle ve kalkış gazı ver.
        try:
            reason_l = str(reason or "").lower()
            red_approach_bootstrap = (
                "red_light_" in reason_l
                and not self.is_red_light_stop_reason(reason)
                and "decision_slow" in reason_l
            )
            green_release_force_active = (
                bool(getattr(self, "tl_green_release_active", False))
                and bool(getattr(self, "tl_green_release_from_hold", False))
                and time.time() < float(getattr(self, "green_release_force_until", 0.0) or 0.0)
                and (
                    self.latest_tl_state() == "green"
                    or "green_light_confirmed_stable" in reason_l
                    or "tl_green_release_no_stopline_hold" in reason_l
                    or "tl_no_stopline_hold_go_release" in reason_l
                    or "tl_green_smooth_release" in reason_l
                )
            )

            if (
                float(target_speed) > 0.30
                and float(current_speed) < 0.20
                and ("decision_go" in reason_l or red_approach_bootstrap or green_release_force_active)
                and "timeout" not in reason_l
                and not bool(getattr(self, "tl_stop_hold_active", False))
                and not bool(getattr(self, "red_hold_active", False))
                and "green_wait_stable" not in reason_l
                and "decision_hold_after_stop" not in reason_l
            ):
                if red_approach_bootstrap:
                    if "crawl" in reason_l:
                        min_start_throttle = 0.14
                    else:
                        min_start_throttle = 0.26
                    tag = "standstill_red_approach_bootstrap"
                else:
                    min_start_throttle = 0.24 if green_release_force_active else 0.38
                    tag = "standstill_go_bootstrap"

                if green_release_force_active:
                    min_start_throttle = min(min_start_throttle, 0.24)

                control.throttle = max(float(getattr(control, "throttle", 0.0) or 0.0), min_start_throttle)
                if green_release_force_active:
                    control.throttle = min(float(control.throttle), 0.32)
                control.brake = 0.0
                reason = str(reason) + "|" + tag
                self.last_throttle_cmd = float(control.throttle)
                self.last_brake_cmd = 0.0
        except Exception:
            pass

        if (
            bool(getattr(self, "tl_waiting_for_verified_green", False))
            and not self.tl_wait_green_release_allowed(reason)
        ):
            control.throttle = 0.0
            control.brake = 1.0
            control.hand_brake = True
            control.reverse = False
            control.manual_gear_shift = False
            target_speed = 0.0
            self.red_hold_active = True
            self.red_hold_until_green = True
            self.tl_stop_hold_active = True
            self.tl_stop_hold_until = time.time() + 45.0
            if "tl_stationary_hold_no_rollback" not in str(reason):
                reason = self.wait_green_latch_reason(reason, getattr(self, "last_tl_stopline_debug", {}) or {})
            self.last_throttle_cmd = 0.0
            self.last_brake_cmd = 1.0

        self.ego.apply_control(control)

        target = self.get_target() or {}
        mission_dist = self.latest_mission.get("distance_to_target_m") if self.latest_mission else None

        payload = {
            "stamp": time.time(),
            "mission_stage": self.get_stage(),
            "task_index": self.latest_mission.get("task_index") if self.latest_mission else None,
            "target_name": target.get("name"),
            "target_is_planner_local": self.is_planner_local_target(target),
            "planner_local_as_destination": self.planner_local_target_as_destination,
            "distance_to_target_m": mission_dist,
            "target_speed_mps": round(target_speed, 3),
            "target_speed_kmh": round(self.mps_to_kmh(target_speed), 1),
            "current_speed_mps": round(current_speed, 3),
            "current_speed_kmh": round(self.mps_to_kmh(current_speed), 1),
            "throttle": round(control.throttle, 3),
            "brake": round(control.brake, 3),
            "steer": round(control.steer, 3),
            "route_status": self.route_status,
            "driver_only_decision_mode": bool(getattr(self, "driver_only_decision_mode", False)),
            "planner_enabled": self.use_planner_local_target,
            "planner_target_fresh": self.get_fresh_planner_target() is not None,
            "planner_target_age_s": round(time.time() - self.last_local_target_time, 3)
            if self.latest_local_target is not None else None,
            "planner_route_id": self.latest_local_target.get("route_id")
            if isinstance(self.latest_local_target, dict) else None,
            "lane": self.current_lane_debug,
            "selected_stopline_id": (getattr(self, "last_tl_stopline_debug", {}) or {}).get("selected_stopline_id"),
            "line_dist": (getattr(self, "last_tl_stopline_debug", {}) or {}).get("line_dist"),
            "control_dist": (getattr(self, "last_tl_stopline_debug", {}) or {}).get("control_dist"),
            "stop_before": (getattr(self, "last_tl_stopline_debug", {}) or {}).get("stop_before"),
            "lat": (getattr(self, "last_tl_stopline_debug", {}) or {}).get("lat"),
            "red_hold_active": bool(getattr(self, "red_hold_active", False)),
            "passed_ignore": bool(
                (getattr(self, "last_tl_stopline_debug", {}) or {}).get(
                    "passed_ignore",
                    self.selected_stopline_passed_ignore(),
                )
            ),
            "tl_state": self.latest_tl_state(),
            "reason": reason,
        }

        msg = String()
        msg.data = json.dumps(payload, ensure_ascii=False)
        self.debug_pub.publish(msg)

        self.get_logger().info(
            f"[TEKNOFEST ROUTE] stage={payload['mission_stage']} "
            f"target_name={payload['target_name']} "
            f"dist={payload['distance_to_target_m']} "
            f"target={target_speed:.2f}mps/{self.mps_to_kmh(target_speed):.1f}kmh "
            f"speed={current_speed:.2f}mps/{self.mps_to_kmh(current_speed):.1f}kmh "
            f"throttle={control.throttle:.2f} brake={control.brake:.2f} "
            f"steer={control.steer:.2f} lane={self.current_lane_debug.get('reason')} route={self.route_status} "
            f"selected_stopline_id={payload['selected_stopline_id']} "
            f"line_dist={payload['line_dist']} control_dist={payload['control_dist']} "
            f"stop_before={payload['stop_before']} lat={payload['lat']} "
            f"red_hold_active={payload['red_hold_active']} passed_ignore={payload['passed_ignore']} "
            f"tl_state={payload['tl_state']} "
            f"reason={reason}",
            throttle_duration_sec=0.5,
        )


def main(args=None):
    rclpy.init(args=args)
    node = TeknofestRouteAgentNode()

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
