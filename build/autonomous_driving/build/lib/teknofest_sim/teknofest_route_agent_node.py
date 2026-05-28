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
        self.declare_parameter("green_release_min_start_speed_mps", 3.0)
        self.declare_parameter("red_light_hard_stop_enabled", True)
        self.declare_parameter("red_light_hard_stop_handbrake_below_mps", 0.20)

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
        self.red_light_hard_stop_enabled = bool(
            self.get_parameter("red_light_hard_stop_enabled").value
        )
        self.red_light_hard_stop_handbrake_below_mps = float(
            self.get_parameter("red_light_hard_stop_handbrake_below_mps").value
        )

        self.manual_tl_stoplines_enabled = bool(
            self.get_parameter("manual_tl_stoplines_enabled").value
        )
        self.manual_tl_stoplines_path = str(
            self.get_parameter("manual_tl_stoplines_path").value
        )

        # TL_HOLD_SAFE_GO_RELEASE_FIX:
        # Kırmızı HOLD sonrası perception green'i kaçırırsa,
        # decision birkaç saniye stabil GO verince kontrollü release.
        self.tl_hold_safe_go_release_s = 5.0
        self.tl_hold_go_candidate_since = 0.0
        self.tl_post_green_ignore_until = 0.0
        self.tl_stop_hold_active = False
        self.tl_stop_hold_until = 0.0
        self.tl_hold_last_update = 0.0
        self.tl_hold_source = None
        self.tl_hold_stopline_id = None
        self.tl_hold_route_valid = False
        self.tl_hold_last_clear_reason = None
        self.green_release_force_until = 0.0

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
            new_stop_reason = str(data.get("stop_reason", "") or "").lower()
            new_release_condition = str(data.get("release_condition", "") or "").lower()
            new_should_stop = bool(data.get("should_stop", False))

            # TL_HOLD_SAFE_GO_RELEASE_FIX:
            # Kırmızıda durduktan sonra bazı frame'lerde yeşil görünse bile
            # perception/decision green_light_confirmed_stable üretemeyebiliyor.
            # Eğer decision stabil GO veriyor ve yeni red/yellow yoksa sayaç başlat.
            go_candidate = (
                new_decision == "GO"
                and new_state not in {"red", "yellow"}
                and "red_light" not in new_reason
                and "yellow_light" not in new_reason
            )

            if go_candidate:
                if float(getattr(self, "tl_hold_go_candidate_since", 0.0) or 0.0) <= 0.0:
                    self.tl_hold_go_candidate_since = now
            else:
                self.tl_hold_go_candidate_since = 0.0

            # TL_STOP_HOLD_FIX:
            # Kırmızıda araç tam durmuşsa, ışık görüntüden kayboldu/unknown oldu diye
            # tekrar yürümemeli. Sadece doğrulanmış yeşil bunu temizler.
            # TL_STRICT_GREEN_RELEASE_STATE_MACHINE_FIX:
            # HOLD sadece confirmed/stable green ile açılır.
            # traffic_light_state == "green" tek başına yetmez;
            # çünkü green_wait_stable:1/3 veya 2/3 sırasında da state green geliyor.
            # Enforce strict semantics: only the explicit release_condition
            # == "confirmed_green" clears TL holds. Remove heuristic
            # string matching fallbacks to avoid accidental releases.
            green_release_confirmed = (new_release_condition == "confirmed_green")

            if green_release_confirmed:
                self.tl_stop_hold_active = False
                self.tl_stop_hold_until = 0.0
                self.tl_hold_last_update = 0.0
                self.tl_hold_source = "confirmed_green_release"
                self.tl_hold_route_valid = False

                self.green_release_force_until = now + 2.0
                # GREEN_RELEASE_POST_IGNORE_FIX:
                # Yeşilden kalktıktan sonra aynı kavşaktaki/yan şerit ışıkları
                # red/yellow görünse bile aracı tekrar kilitlemesin.
                self.tl_post_green_ignore_until = now + 4.0
                self.last_brake_cmd = 0.0
                self.last_throttle_cmd = 0.0

            tl_structured_stop_now = (
                new_should_stop
                and (
                    "traffic_light" in new_stop_reason
                    or "tl_" in new_reason
                    or "unknown_light_precaution_near_route_stopline" in new_reason
                    or "tl_unknown_near_stopline_hold" in new_reason
                    or "tl_stopline_hold_active" in new_reason
                )
            )

            red_stop_now = (
                (
                    "red_light" in new_reason
                    or "yellow_light" in new_reason
                    or "tl_hard_stop" in new_reason
                    or "tl_stopline_control" in new_reason
                )
                and (
                    new_decision == "STOP"
                    or "decision_stop" in new_reason
                    or "hard_stop" in new_reason
                    or "overhead_stop" in new_reason
                    or "manual_tl_stopline_stop" in new_reason
                )
            )

            hold_ctx = self.extract_tl_hold_context(data, self.get_fresh_planner_target())
            if red_stop_now or tl_structured_stop_now:
                if not hold_ctx["route_valid"]:
                    self.tl_stop_hold_active = False
                    self.tl_stop_hold_until = 0.0
                    self.tl_hold_last_update = 0.0
                    self.tl_hold_source = "invalid_stopline_rejected"
                    self.tl_hold_route_valid = False
                    self.tl_hold_last_clear_reason = "TL_HOLD_CLEARED_BECAUSE_INVALID_STOPLINE"
                    self.latest_decision = data
                    self.last_decision_time = now
                    return
                self.tl_stop_hold_active = True
                # Kırmızı döngü uzun sürerse bile güvenli tarafta kal.
                # Yeşil confirmed geldiğinde zaten hemen temizlenir.
                self.tl_stop_hold_until = now + 45.0
                self.tl_hold_last_update = now
                self.tl_hold_source = "decision_tl_stop"
                self.tl_hold_stopline_id = hold_ctx["active_stopline_id"]
                self.tl_hold_route_valid = bool(hold_ctx["route_valid"])

            self.latest_decision = data
            self.last_decision_time = now
        except Exception as exc:
            self.get_logger().warning(f"decision parse hatası: {exc}")

    def extract_tl_hold_context(self, decision=None, planner_target=None):
        decision = decision if isinstance(decision, dict) else (getattr(self, "latest_decision", {}) or {})
        planner_target = planner_target if isinstance(planner_target, dict) else None
        ctx = {}
        route_intent = decision.get("route_intent") if isinstance(decision.get("route_intent"), dict) else {}
        if isinstance(route_intent.get("stopline_context"), dict):
            ctx.update(route_intent.get("stopline_context"))
        if isinstance(decision.get("stopline_context"), dict):
            ctx.update(decision.get("stopline_context"))
        if planner_target and isinstance(planner_target.get("stopline_context"), dict):
            target_ctx = planner_target.get("stopline_context")
            if not ctx or target_ctx.get("stopline_id") == ctx.get("stopline_id"):
                ctx.update(target_ctx)

        active_id = (
            decision.get("stopline_id")
            or ctx.get("stopline_id")
            or ctx.get("id")
            or (planner_target or {}).get("stopline_id")
        )
        route_valid = bool(
            decision.get(
                "stopline_route_valid",
                ctx.get("route_valid", (planner_target or {}).get("stopline_route_valid", False)),
            )
        )
        return {
            "active_stopline_id": active_id,
            "route_valid": route_valid,
            "road_match": bool(ctx.get("road_match", decision.get("stopline_road_match", False))),
            "lane_match": bool(ctx.get("lane_match", decision.get("stopline_lane_match", False))),
            "route_connected_match": bool(ctx.get("route_connected_match", decision.get("stopline_route_connected_match", False))),
            "selection_reject_reason": ctx.get("selection_reject_reason"),
        }

    def valid_tl_hold_active(self, decision, planner_target, now):
        if not bool(getattr(self, "tl_stop_hold_active", False)):
            return False, self.extract_tl_hold_context(decision, planner_target), "hold_inactive", None

        ctx = self.extract_tl_hold_context(decision, planner_target)
        hold_age = now - float(getattr(self, "tl_hold_last_update", 0.0) or 0.0)
        sm_state = str((decision or {}).get("tl_state_machine_state", "") or "")
        tl_state = str((decision or {}).get("traffic_light_state", "") or "").lower()
        decision_name = str((decision or {}).get("decision", "") or "").upper()
        should_stop = bool((decision or {}).get("should_stop", False))
        hold_id = getattr(self, "tl_hold_stopline_id", None)
        active_id = ctx.get("active_stopline_id")
        same_stopline = bool(hold_id and active_id and str(hold_id) == str(active_id))
        state_allows_hold = sm_state in {"BRAKING_TO_STOPLINE", "STOPPED_AT_STOPLINE", "WAITING_FOR_GREEN"}
        risk_light = tl_state in {"red", "yellow"} or (tl_state == "unknown" and should_stop)

        valid = (
            hold_age <= 1.5
            and bool(ctx.get("route_valid"))
            and same_stopline
            and state_allows_hold
            and (risk_light or should_stop)
        )
        reason = "valid" if valid else (
            "stale_hold" if hold_age > 1.5 else
            "route_invalid" if not ctx.get("route_valid") else
            "stopline_mismatch" if not same_stopline else
            "state_not_hold" if not state_allows_hold else
            "decision_go_no_risk" if decision_name == "GO" and not should_stop else
            "not_risk_light"
        )
        return valid, ctx, reason, hold_age

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
            debug["reason"] = "LANE_KEEP_FALLBACK_ROUTE_CENTER:lane_timeout"
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
            debug["reason"] = "LANE_KEEP_FALLBACK_ROUTE_CENTER:lane_not_detected"
            self.current_lane_debug = debug
            return basic_steer

        if conf < self.lane_min_confidence:
            debug["reason"] = f"LANE_KEEP_FALLBACK_ROUTE_CENTER:low_conf:{conf:.3f}"
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
        # MANUAL_TL_STOPLINE_FIX:
        # Önceki patchlerde tick() içinde self.hard_stop_control() çağrısı var,
        # ama metod yoksa route_agent çöküyor. Kırmızı ışık STOP için güvenli fren.
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
        extent_x = 2.25

        try:
            extent_x = float(self.ego.bounding_box.extent.x)
        except Exception:
            pass

        front_x = float(loc.x) + math.cos(yaw) * extent_x
        front_y = float(loc.y) + math.sin(yaw) * extent_x
        return front_x, front_y

    def manual_tl_red_active(self):
        d = getattr(self, "latest_decision", {}) or {}

        state = str(d.get("traffic_light_state", "") or "").lower()
        reason = str(d.get("reason", "") or "").lower()
        latched = bool(d.get("red_light_latch_active", False))

        # Kesin yeşil onayı varsa red stopline kapalı.
        if "green_light_confirmed_stable" in reason:
            return False

        # Sadece state=green ama reason red değilse kapat.
        # Bazı geçiş frame'lerinde state/reason çakışabiliyor; red reason daha güvenli.
        if state == "green" and "red_light" not in reason and not latched:
            return False

        return (
            state == "red"
            or latched
            or "red_light" in reason
        )

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

            # Kritik değişiklik:
            # Stopline yaw yerine aracın anlık yönüne göre mesafe çıkar.
            # Yokuş/kavşaklarda stopline yaw ile eşleşme çok kolay kaçıyor.
            longitudinal_m = dx * float(fwd.x) + dy * float(fwd.y)
            lateral_m = dx * float(right.x) + dy * float(right.y)

            release_after = float(line.get("release_after_pass_m", 4.0))
            approach_m = float(line.get("approach_m", 120.0))

            # STOP_2M_BEFORE_LINE_FIX:
            # Negatif longitudinal_m, ön tamponun bu stopline'ı geçtiği anlamına gelir.
            # Bu çizgiyi yeni hard-stop sebebi yaparsak araç çizgiyi geçtikten sonra frenler.
            if longitudinal_m < 0.30:
                continue
            if longitudinal_m > approach_m:
                continue

            yaw_diff = _angle_diff(line.get("yaw_deg", 0.0), ego_yaw)

            line_road = line.get("road_id", None)
            line_lane = line.get("lane_id", None)

            same_lane = False
            try:
                same_lane = (
                    ego_road is not None
                    and ego_lane is not None
                    and line_road is not None
                    and line_lane is not None
                    and int(line_road) == int(ego_road)
                    and int(line_lane) == int(ego_lane)
                )
            except Exception:
                same_lane = False

            candidates.append({
                "line": line,
                "dist": float(longitudinal_m),
                "lat": float(lateral_m),
                "yaw_diff": float(yaw_diff),
                "same_lane": bool(same_lane),
            })

        if not candidates:
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

        # PASS-1: sıkı eşleşme.
        strict = []
        for c in candidates:
            lat_limit = float(c["line"].get("lateral_half_width_m", 6.0))
            if c["same_lane"]:
                lat_limit = max(lat_limit, 6.5)

            if abs(c["lat"]) <= lat_limit and (c["same_lane"] or c["yaw_diff"] <= 95.0):
                score = abs(c["lat"]) + c["dist"] * 0.020 + c["yaw_diff"] * 0.004
                if c["same_lane"]:
                    score -= 7.0
                strict.append((score, c))

        if strict:
            _, c = min(strict, key=lambda x: x[0])
            return c["line"], c["dist"], c["lat"]

        # PASS-2:
        # Yokuş ve kavşakta road/lane/yaw şaşabiliyor.
        # Önde, enine makul yakın stopline'ı fallback kabul et.
        loose = []
        for c in candidates:
            if abs(c["lat"]) <= 12.0 and c["yaw_diff"] <= 150.0:
                score = abs(c["lat"]) + c["dist"] * 0.030 + c["yaw_diff"] * 0.002
                loose.append((score, c))

        if loose:
            _, c = min(loose, key=lambda x: x[0])
            return c["line"], c["dist"], c["lat"]

        # PASS-3: yokuş/kavşak acil yakın stopline fallback.
        # CARLA stop waypoint yaw/road/lane bazen kavşakta 90/180 derece ve same_lane=0 geliyor.
        # Logda gerçek yakın adaylar lat=12.9/13.0 ile kaçıyordu.
        # Çok yakındaki stopline'ı, yaw'a bakmadan kabul et; aksi halde visual hard-stop geç kalıyor.
        emergency = []
        for c in candidates:
            d = float(c["dist"])
            lat = abs(float(c["lat"]))
            if 0.0 <= d <= 18.0 and lat <= 16.0:
                # Yakın mesafeyi lateralden daha önemli yap.
                score = d * 0.60 + lat * 0.40
                emergency.append((score, c))

        if emergency:
            _, c = min(emergency, key=lambda x: x[0])
            return c["line"], c["dist"], c["lat"]

        return None


    def apply_manual_tl_stopline_cap(self, target_speed, reason):
        if not getattr(self, "manual_tl_stoplines_enabled", False):
            return target_speed, reason

        stage = str(self.get_stage() or "")
        if stage == "PASSENGER_STOP":
            return target_speed, reason

        reason_l = str(reason or "").lower()
        current_red_yellow = (
            "red_light" in reason_l
            or "yellow_light" in reason_l
            or "traffic_light_red" in reason_l
            or "traffic_light_yellow" in reason_l
        )

        if not self.manual_tl_red_active() and not current_red_yellow:
            return target_speed, reason

        hit = self.nearest_manual_tl_stopline()
        if hit is None:
            dbg = str(getattr(self, "_last_manual_tl_candidates_debug", "") or "no_debug")
            return target_speed, f"{reason}|manual_tl_stopline_no_match:{dbg}"

        line, dist_m, lat_m = hit
        line_id = str(line.get("id", "manual_stopline"))

        # STOP_2M_BEFORE_LINE_FIX:
        # dist_m = ön tampondan CARLA stop waypoint'e kalan mesafe.
        # İstenen davranış: çizginin üstünde değil, çizgiden 2m önce dur.
        stop_before_m = float(line.get("stop_before_m", 1.00))
        control_dist_m = float(dist_m) - stop_before_m

        # Bu satıra normalde düşmemeli çünkü nearest negatifleri eleyecek.
        # Yine de güvenlik için: geçmiş çizgiyi yeni hold sebebi yapma.
        if dist_m < 0.30:
            return target_speed, (
                f"{reason}|manual_tl_stopline_passed_ignore:"
                f"id={line_id},line_dist={dist_m:.2f},lat={lat_m:.2f}"
            )

        stop_margin = 0.15
        crawl_m_base = float(line.get("crawl_m", 14.0))
        crawl_speed = float(line.get("crawl_speed_mps", 1.10))
        approach_speed = float(line.get("approach_speed_mps", 4.00))

        try:
            base_speed = float(target_speed or 0.0)
        except Exception:
            base_speed = 0.0

        try:
            current_speed = float(self.get_speed())
        except Exception:
            current_speed = 0.0

        v = max(0.0, current_speed)

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

        # Artık STOP kararı line_dist değil, control_dist üzerinden.
        # control_dist <= 0: çizgiden 2m önceki hedefe geldik.
        emergency_stop_m = max(stop_margin, min(6.0, 1.0 + 0.45 * v))

        if control_dist_m <= max(stop_margin, emergency_stop_m):
            return 0.0, (
                f"{reason}|red_light_manual_tl_stopline_stop:"
                f"id={line_id},line_dist={dist_m:.2f},control_dist={control_dist_m:.2f},"
                f"stop_before={stop_before_m:.2f},lat={lat_m:.2f},v={v:.2f}"
            )

        if control_dist_m <= dynamic_crawl_m:
            capped = choose_speed(crawl_speed)
            if control_dist_m <= 5.0:
                capped = min(capped, 0.65)

            return capped, (
                f"{reason}|red_light_manual_tl_stopline_crawl:"
                f"id={line_id},line_dist={dist_m:.2f},control_dist={control_dist_m:.2f},"
                f"stop_before={stop_before_m:.2f},lat={lat_m:.2f},cap={capped:.2f},"
                f"v={v:.2f},brake_dist={brake_distance:.1f}"
            )

        capped = choose_speed(approach_speed)
        return capped, (
            f"{reason}|red_light_manual_tl_stopline_approach:"
            f"id={line_id},line_dist={dist_m:.2f},control_dist={control_dist_m:.2f},"
            f"stop_before={stop_before_m:.2f},lat={lat_m:.2f},cap={capped:.2f},"
            f"v={v:.2f},brake_dist={brake_distance:.1f}"
        )


    def reset_longitudinal_memory(self):
        # TL_CRASH_FIX:
        # Kırmızı STOP / hard_stop sonrası throttle-brake hafızasını temizle.
        # Bu metod tick() içinde çağrılıyor; yoksa route_agent crash oluyor.
        self.last_throttle_cmd = 0.0
        self.last_brake_cmd = 0.0

        try:
            self.last_smoothed_target_speed_mps = 0.0
            self.last_smoothed_target_time = time.time()
        except Exception:
            pass

    def apply_planner_speed_hint_cap(self, target_speed, reason):
        # TL_STOP_HOLD_FIX:
        # Kırmızıda tam durduktan sonra perception unknown/GO'ya düşse bile
        # confirmed green gelmeden aracı tekrar yürütme.
        now = time.time()
        reason_l = str(reason or "").lower()

        # UPCOMING_TL_UNCERTAIN_STOPLINE_PRECAP_FIX:
        # Yokuşta kırmızı/yellow çok geç gelebiliyor. Eğer confirmed green yoksa
        # ve önde manual stopline varsa 30 km/h ile kavşağa dalma.
        try:
            latest_decision = getattr(self, "latest_decision", {}) or {}
            latest_state = str(latest_decision.get("traffic_light_state", "") or "").lower()
            latest_reason = str(latest_decision.get("reason", "") or "").lower()

            # Only consider confirmed when the decision explicitly carries
            # the release condition. Do not use legacy reason string matching.
            latest_release = str(latest_decision.get("release_condition", "") or "").lower()
            confirmed_green_now = (latest_release == "confirmed_green")

            has_current_red_yellow = (
                "red_light" in reason_l
                or "yellow_light" in reason_l
                or "traffic_light_red" in reason_l
                or "traffic_light_yellow" in reason_l
            )

            # Sadece belirsiz durumda pre-cap. Red/yellow varsa manual stopline bloğu zaten yönetecek.
            if (
                not confirmed_green_now
                and not has_current_red_yellow
                and float(target_speed or 0.0) > 4.05
                and latest_state not in {"red", "yellow"}
            ):
                hit = self.nearest_manual_tl_stopline()
                if hit is not None:
                    line, dist_m, lat_m = hit
                    stop_before_m = float(line.get("stop_before_m", 1.0))
                    control_dist_m = float(dist_m) - stop_before_m

                    try:
                        v = float(self.get_speed())
                    except Exception:
                        v = 0.0

                    if 0.0 < control_dist_m <= 55.0:
                        cap = 4.0
                        if control_dist_m <= 28.0:
                            cap = 3.0
                        if control_dist_m <= 16.0:
                            cap = 2.2
                        if control_dist_m <= 8.0:
                            cap = 1.2

                        target_speed = min(float(target_speed or 0.0), cap)
                        reason = (
                            f"{reason}|upcoming_tl_uncertain_stopline_precap:"
                            f"id={str(line.get('id','?'))},line_dist={float(dist_m):.2f},"
                            f"control_dist={control_dist_m:.2f},stop_before={stop_before_m:.2f},"
                            f"lat={float(lat_m):.2f},cap={target_speed:.2f},v={v:.2f}"
                        )
                        reason_l = str(reason or "").lower()
        except Exception as exc:
            try:
                self.get_logger().warning(f"upcoming TL uncertain pre-cap failed: {exc}")
            except Exception:
                pass

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

        # DOWNHILL_WEAK_GREEN_STOPLINE_PRECAUTION_FIX:
        # Yokuşta bazen ilgili kırmızı geç/yanlış algılanıyor; önümüzde manual stopline
        # varsa ve ışık henüz confirmed green değilse tam hızla dalma, kontrollü yaklaş.
        try:
            latest_decision = getattr(self, "latest_decision", {}) or {}
            latest_state = str(latest_decision.get("traffic_light_state", "") or "").lower()
            latest_reason = str(latest_decision.get("reason", "") or "").lower()

            confirmed_green = (
                "green_light_confirmed_stable" in reason_l
                or "green_light_confirmed_stable" in latest_reason
            )

            weak_or_unconfirmed_green = (
                latest_state == "green"
                and not confirmed_green
                and not tl_hold_active_now
            )

            if weak_or_unconfirmed_green:
                hit = self.nearest_manual_tl_stopline()
                if hit is not None:
                    line, dist_m, lat_m = hit

                    try:
                        v = float(self.get_speed())
                    except Exception:
                        v = 0.0

                    # Dinamik fren mesafesi. v=8 m/s için yaklaşık 22-25m bandı.
                    decel = 2.35
                    reaction_s = 0.75
                    brake_dist = (v * v) / max(0.1, 2.0 * decel)
                    brake_dist += reaction_s * v
                    brake_dist += 5.0

                    try:
                        pitch = float(self.ego.get_transform().rotation.pitch)
                        if abs(pitch) >= 2.5:
                            brake_dist += 4.0
                    except Exception:
                        pass

                    # Stopline yaklaşıyorsa ama yeşil henüz confirmed değilse,
                    # full stop değil, güvenli yaklaşma.
                    if 0.0 < float(dist_m) <= max(22.0, brake_dist):
                        cap = 4.0
                        if float(dist_m) <= 12.0:
                            cap = 2.2
                        if float(dist_m) <= 6.0:
                            cap = 1.2

                        try:
                            current_target = float(target_speed or 0.0)
                        except Exception:
                            current_target = 0.0

                        if current_target > 0.05:
                            cap = min(current_target, cap)

                        return cap, (
                            f"{reason}|weak_green_stopline_precaution:"
                            f"id={str(line.get('id','?'))},dist={float(dist_m):.2f},"
                            f"lat={float(lat_m):.2f},cap={cap:.2f},v={v:.2f},brake_dist={brake_dist:.1f}"
                        )

        except Exception as exc:
            try:
                self.get_logger().warning(f"weak green stopline precaution failed: {exc}")
            except Exception:
                pass

        # GREEN_RELEASE_POST_IGNORE_FIX:
        # Confirmed green ile kalktıktan sonra aynı kavşağın eski red/yellow'u aracı
        # tekrar kilitlemesin. Ancak önümüzde yeni/manual stopline varsa ASLA ignore etme;
        # aksi halde yakın ikinci kırmızıda geçme yapar.
        red_yellow_reason = (
            "yellow_light_detected" in reason_l
            or "red_light_no_sensor" in reason_l
            or "red_light_" in reason_l
        )

        manual_stopline_ahead = False
        if red_yellow_reason:
            try:
                manual_stopline_ahead = self.nearest_manual_tl_stopline() is not None
            except Exception:
                manual_stopline_ahead = False

        if (
            now < float(getattr(self, "tl_post_green_ignore_until", 0.0) or 0.0)
            and red_yellow_reason
            and not manual_stopline_ahead
            and "green_light_confirmed_stable" not in reason_l
        ):
            self.tl_stop_hold_active = False
            self.tl_stop_hold_until = 0.0
            self.last_brake_cmd = 0.0
            return max(float(target_speed or 0.0), 2.0), f"{reason}|post_green_same_intersection_tl_ignore"

        # TL_CARLA_STOPLINE_PRIORITY_FIX:
        # CARLA/manual stopline varsa bbox'a göre erken/geç hard-stop yapma.
        # Önce dünya koordinatındaki stopline'a yaklaş/crawl/stop uygula.
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

                if "red_light_manual_tl_stopline_stop" in reason_s:
                    self.tl_stop_hold_active = True
                    self.tl_stop_hold_until = now + 45.0
                    self.last_brake_cmd = 0.0
                    self.last_throttle_cmd = 0.0
                    return 0.0, f"{reason}|red_light_hard_stop"

                if "red_light_manual_tl_stopline_" in reason_s:
                    return float(target_speed), f"{reason}|skip_visual_bbox_stop:manual_stopline_active"

        except Exception as exc:
            try:
                self.get_logger().warning(f"manual stopline priority failed: {exc}")
            except Exception:
                pass

        # TL_RED_CRAWL_FORCE_STOP_FIX:
        # TL_RED_FAR_SLOW_EARLY_FORCE_STOP_FIX:
        # Yokuş kırmızısında decision bazen STOP yerine
        # red_light_no_sensor_crawl_visual ile 4 km/h SLOW veriyor.
        # Bu, özellikle yokuş aşağı durumda çizgiyi kaçırıyor.
        # Eğer kırmızı ışık bbox'u artık yeterince büyükse bunu route_agent seviyesinde
        # kesin STOP'a çeviriyoruz. Uzak/minik kırmızılar bu eşikleri geçmez.
        if (
            "red_light_no_sensor_crawl_visual" in reason_l
            or "red_light_no_sensor_far_slow" in reason_l
        ):
            def _metric(key, default=None):
                try:
                    text = str(reason or "")
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

            b = _metric("b")
            h = _metric("h")
            a = _metric("a")

            # Eski yanlış uzak kırmızı örneği: h≈0.026, a≈0.00018
            # Yokuştaki gerçek kırmızı artık h≈0.050 veya a≈0.00044 seviyesinde
            # far_slow'dan STOP'a alınmalı. Yoksa 20 km/h yaklaşma geç frenletiyor.
            red_crawl_should_stop = (
                (h is not None and h >= 0.050)
                or (a is not None and a >= 0.00044)
                or (
                    "red_light_no_sensor_crawl_visual" in reason_l
                    and (
                        (h is not None and h >= 0.050)
                        or (a is not None and a >= 0.00055)
                    )
                )
            )

            if red_crawl_should_stop:
                self.tl_stop_hold_active = True
                self.tl_stop_hold_until = now + 45.0
                self.last_brake_cmd = 0.0
                self.last_throttle_cmd = 0.0
                return 0.0, (
                    f"{reason}|red_light_visual_forced_stop:"
                    f"b={b},h={h},a={a}|red_light_hard_stop"
                )

        latest_state = ""
        latest_reason = ""
        try:
            latest_state = str((getattr(self, "latest_decision", {}) or {}).get("traffic_light_state", "") or "").lower()
            latest_reason = str((getattr(self, "latest_decision", {}) or {}).get("reason", "") or "").lower()
        except Exception:
            pass

        # TL_STRICT_GREEN_RELEASE_STATE_MACHINE_FIX:
        # Only treat release as confirmed when the decision explicitly
        # indicates `release_condition == 'confirmed_green'`.
        latest_release = str((getattr(self, "latest_decision", {}) or {}).get("release_condition", "") or "").lower()
        green_confirmed = (latest_release == "confirmed_green" or str(reason or "").lower() == "confirmed_green")

        if green_confirmed:
            self.tl_stop_hold_active = False
            self.tl_stop_hold_until = 0.0

        if (
            bool(getattr(self, "tl_stop_hold_active", False))
            and now < float(getattr(self, "tl_stop_hold_until", 0.0) or 0.0)
            and not green_confirmed
        ):
            # TL_HOLD_SAFE_GO_RELEASE_FIX:
            # Confirmed green gelmediyse ama decision stabil GO diyorsa ve
            # yeni red/yellow yoksa, yeşil algısı kaçmış kabul edip kontrollü release et.
            go_since = float(getattr(self, "tl_hold_go_candidate_since", 0.0) or 0.0)
            go_age = now - go_since if go_since > 0.0 else 0.0

            # TL_STRICT_GREEN_RELEASE_STATE_MACHINE_FIX:
            # Safe release kapalı. Kırmızı HOLD aktifken GO/unknown aracı kaldıramaz.
            # Release sadece green_light_confirmed_stable ile olur.
            safe_go_release = False

            return 0.0, f"{reason}|tl_stop_hold_until_confirmed_green|red_light_hard_stop"
        target_speed, reason = self.apply_manual_tl_stopline_cap(target_speed, reason)
        if "red_light_manual_tl_stopline_stop" in str(reason):
            return target_speed, reason

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
            "green_light_confirmed_stable" in reason_l
            or time.time() < float(getattr(self, "green_release_force_until", 0.0) or 0.0)
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

        # TL_STOPLINE_NO_UPWARD_TARGET_SMOOTH_FIX:
        # Stopline / weak-green önlemi düşük hız cap'i verdiyse smoothing bunu yukarı çekemez.
        # Önceki logda weak_green_stopline_precaution cap=2.20 iken target_smooth 7.28'e çekmişti.
        tl_stopline_hard_cap_active = (
            "red_light_manual_tl_stopline_" in reason_l
            or "weak_green_stopline_precaution" in reason_l
            or "skip_visual_bbox_stop:manual_stopline_active" in reason_l
            or "upcoming_tl_uncertain_stopline_precap" in reason_l
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

        lane_reason = "no_auto_right_shift"
        lane_wp = wp
        stage_l = stage.lower()
        maneuver = str(
            target.get("route_intent")
            or target.get("current_maneuver")
            or target.get("road_option")
            or ""
        ).lower()

        if (
            self.is_planner_local_target(target)
            and "park" not in stage_l
            and maneuver not in {"left", "turn_left"}
        ):
            try:
                base_sign = 1 if int(wp.lane_id) > 0 else -1
                cur_wp = wp
                hops = 0
                while hops < 3:
                    right_wp = cur_wp.get_right_lane()
                    if right_wp is None:
                        break
                    if right_wp.lane_type != self.carla.LaneType.Driving:
                        break
                    try:
                        if int(right_wp.lane_id) * base_sign <= 0:
                            break
                    except Exception:
                        break
                    lane_wp = right_wp
                    cur_wp = right_wp
                    hops += 1
                if lane_wp is not wp:
                    lane_reason = "LANE_KEEP_RIGHT:rightmost_same_direction"
            except Exception:
                lane_wp = wp

        loc = lane_wp.transform.location
        dest = self.carla.Location(x=loc.x, y=loc.y, z=loc.z + 0.2)

        self.get_logger().info(
            f"Destination lane approach SAFE_CENTER: target={target.get('name')} "
            f"stage={stage} shift=0.00 reason={lane_reason} "
            f"lane_id={getattr(lane_wp, 'lane_id', None)} "
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
        red_stop = "red_light_" in text and (
            "stop" in text
            or "visual_stop" in text
            or "no_distance_bottom" in text
        )

        prev_brake = float(getattr(self, "last_brake_cmd", 0.0) or 0.0)
        prev_steer = float(getattr(self, "last_steer_cmd", 0.0) or 0.0)

        control.throttle = 0.0
        control.steer = self.clamp(prev_steer, -0.35, 0.35)

        if red_stop:
            # Kırmızı STOP ise gecikmesiz net fren.
            control.brake = 1.0
        else:
            # Genel STOP için yumuşak ama kararlı fren.
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

    def latest_decision_is_structured(self):
        d = getattr(self, "latest_decision", {}) or {}
        return bool(d.get("command_version") or d.get("target_speed_mps") is not None or d.get("emergency_brake") is not None)

    def apply_planner_speed_hint_only(self, target_speed, reason):
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

        d = getattr(self, "latest_decision", {}) or {}
        if bool(d.get("emergency_brake", False)) or bool(d.get("should_stop", False)):
            return min(target_speed, speed_hint), f"{reason}|planner_speed_hint_cap:{speed_hint:.2f}"

        capped = min(target_speed, speed_hint)
        if capped < target_speed - 0.01:
            reason = f"{reason}|planner_speed_hint_cap:{speed_hint:.2f}"
        return capped, reason

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

        # Passenger pull-over behavior: approach target slowly, then stop and hold.
        if stage == "PASSENGER_STOP":
            # Distance to mission target (may be provided by mission node)
            dist_to_target = None
            try:
                dist_to_target = float((self.latest_mission or {}).get("distance_to_target_m"))
            except Exception:
                dist_to_target = None

            # tolerance for stop (mission node default 1.15m)
            try:
                tol = float((self.latest_mission or {}).get("task_stop_tolerance_m", 1.15))
            except Exception:
                tol = 1.15

            # approach slowly while farther than tolerance
            approach_speed = min(float(getattr(self, "slow_speed_mps", 2.0)), 1.2)
            if dist_to_target is None:
                return approach_speed, f"passenger_stop_approach:dist_unknown"

            if dist_to_target <= max(0.5, tol):
                # reached stop location: request hard stop and record start time
                try:
                    if not hasattr(self, "passenger_stop_started_at") or self.passenger_stop_started_at is None:
                        self.passenger_stop_started_at = time.time()
                except Exception:
                    pass
                return 0.0, f"passenger_stop_target_reached|ROUTE_AGENT_APPLY_STOP"

            return approach_speed, f"passenger_stop_approach:dist={dist_to_target:.2f}"

        if now - self.last_decision_time > 2.0:
            return 0.0, "decision_timeout"

        decision = str(self.latest_decision.get("decision", "STOP")).upper()
        raw_reason = str(self.latest_decision.get("reason", "unknown"))
        structured_should_stop = bool(self.latest_decision.get("should_stop", False))
        structured_emergency = bool(self.latest_decision.get("emergency_brake", False))

        decision_speed = None
        speed_reason = "decision_speed_missing"

        try:
            if self.latest_decision.get("target_speed_mps") is not None:
                decision_speed = float(self.latest_decision.get("target_speed_mps"))
                speed_reason = f"decision_speed_mps:{decision_speed:.2f}"
            elif self.latest_decision.get("target_speed_kmh") is not None:
                decision_speed = self.kmh_to_mps(self.latest_decision.get("target_speed_kmh"))
                speed_reason = f"decision_speed_kmh:{float(self.latest_decision.get('target_speed_kmh')):.1f}"
            elif self.latest_decision.get("target_speed") is not None:
                decision_speed = float(self.latest_decision.get("target_speed"))
                speed_reason = f"decision_speed_mps:{decision_speed:.2f}"
        except Exception:
            decision_speed = None
            speed_reason = "decision_speed_parse_error"

        if structured_should_stop or structured_emergency:
            return 0.0, f"driver_only_structured_should_stop:{raw_reason}|STRUCTURED_SHOULD_STOP_HARD_HOLD"

        if decision == "STOP":
            return 0.0, f"driver_only_decision_stop:{raw_reason}|STRUCTURED_SHOULD_STOP_HARD_HOLD"

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

        structured_decision = self.latest_decision_is_structured()
        if structured_decision:
            target_speed, reason = self.apply_planner_speed_hint_only(target_speed, reason)
        else:
            target_speed, reason = self.apply_planner_speed_hint_cap(target_speed, reason)

        latest_decision = getattr(self, "latest_decision", {}) or {}
        emergency_brake = bool(latest_decision.get("emergency_brake", False))
        should_stop = bool(latest_decision.get("should_stop", False))
        stop_reason = str(latest_decision.get("stop_reason", "") or "").lower()
        decision_name = str(latest_decision.get("decision", "") or "").upper()
        latest_target = self.get_fresh_planner_target()
        now_tick = time.time()
        route_blocked = bool(
            isinstance(latest_target, dict)
            and str(latest_target.get("route_decision_status", "")).lower() == "blocked"
        )
        release_confirmed_green = (
            str(latest_decision.get("release_condition", "") or "").lower() == "confirmed_green"
            or "tl_green_release" in str(reason).lower()
            or "green_light_confirmed_stable" in str(reason).lower()
        )
        stop_distance_m = latest_decision.get("stop_distance_m")
        if stop_distance_m is None:
            stop_distance_m = latest_decision.get("stop_target_dist_m")
        try:
            stop_distance_m = float(stop_distance_m) if stop_distance_m is not None else None
        except Exception:
            stop_distance_m = None
        valid_tl_hold, tl_hold_ctx, tl_hold_invalid_reason, tl_hold_age = self.valid_tl_hold_active(
            latest_decision,
            latest_target,
            now_tick,
        )
        decision_go_clear = (
            decision_name == "GO"
            and not should_stop
            and not emergency_brake
            and not route_blocked
        )
        if decision_go_clear and bool(getattr(self, "tl_stop_hold_active", False)) and not valid_tl_hold:
            self.tl_stop_hold_active = False
            self.tl_stop_hold_until = 0.0
            self.tl_hold_route_valid = False
            self.tl_hold_last_clear_reason = f"TL_HOLD_CLEARED_BECAUSE_DECISION_GO:{tl_hold_invalid_reason}"
            reason = f"{reason}|TL_HOLD_CLEARED_BECAUSE_DECISION_GO:{tl_hold_invalid_reason}"

        reason_l_outer = str(reason).lower()
        tl_control_reason = (
            "tl_stopline_control" in reason_l_outer
            or "tl_hard_stop" in reason_l_outer
            or "tl_unknown_near_stopline_hold" in reason_l_outer
            or "tl_hold_keep" in reason_l_outer
            or "tl_hold_set" in reason_l_outer
            or "tl_red_passed" in reason_l_outer
        ) and "tl_hold_cleared_because_decision_go" not in reason_l_outer
        red_light_control = (not structured_decision) and "red_light_" in reason_l_outer
        tl_caution_control = structured_decision and decision_name == "SLOW" and (
            "tl_approach_to_stopline" in reason_l_outer
            or "tl_hold_approach_to_stopline" in reason_l_outer
            or "tl_no_route_valid_stopline_precaution" in reason_l_outer
            or "tl_brake_to_stopline" in reason_l_outer
            or "traffic_light_" in str(stop_reason).lower()
        )
        safety_stopline_control = structured_decision and (
            emergency_brake
            or should_stop
            or decision_name == "STOP"
            or route_blocked
            or (valid_tl_hold and not release_confirmed_green)
            or "tl_hard_stop" in reason_l_outer
            or "tl_red_passed" in reason_l_outer
            or (tl_control_reason and (should_stop or decision_name == "STOP"))
            or ("red_light" in reason_l_outer and (should_stop or decision_name == "STOP"))
            or ("yellow_light" in reason_l_outer and (should_stop or decision_name == "STOP"))
        )
        # Kırmızı ışıkta hedef hız düşüşünü geciktirme.
        # Aksi halde 20 km/h -> 10 km/h -> 0 geçişi çok geç oluyor.
        if not red_light_control and not safety_stopline_control and not tl_caution_control:
            target_speed, reason = self.apply_target_speed_smoothing(target_speed, reason)

        current_speed = self.get_speed()
        if release_confirmed_green:
            self.tl_stop_hold_active = False
            self.tl_stop_hold_until = 0.0
            if "TL_RELEASE_CONFIRMED_GREEN" not in str(reason):
                reason = f"{reason}|TL_RELEASE_CONFIRMED_GREEN"

        if time.time() < self.collision_until:
            control = self.hard_stop_control()
            reason = "collision_halt|ROUTE_AGENT_APPLY_STOP"
            target_speed = 0.0
        elif emergency_brake:
            control = self.hard_stop_control()
            control.hand_brake = current_speed <= 0.20
            reason = f"{reason}|structured_emergency_brake|ROUTE_AGENT_APPLY_STOP"
            self.reset_longitudinal_memory()
        elif structured_decision and (
            should_stop
            or decision_name == "STOP"
            or route_blocked
            or (valid_tl_hold and not release_confirmed_green)
        ):
            # Enforce Decision contract: when Decision explicitly requests stop,
            # set an internal TL hold so route-agent doesn't reintroduce motion.
            control = self.hard_stop_control()
            control.hand_brake = current_speed <= 0.20
            target_speed = 0.0
            try:
                self.tl_stop_hold_active = True
                self.tl_stop_hold_until = now + 45.0
                self.tl_hold_source = "decision_structured"
                self.tl_hold_stopline_id = latest_decision.get("stopline_id")
            except Exception:
                pass
            if should_stop or decision_name == "STOP" or route_blocked:
                reason = f"{reason}|STRUCTURED_SHOULD_STOP_HARD_HOLD|ROUTE_AGENT_APPLY_STOP"
            else:
                reason = (
                    f"{reason}|TL_HOLD_DRIVER_OVERRIDE_STOP:"
                    f"hold_source={getattr(self, 'tl_hold_source', None)},"
                    f"hold_age_s={tl_hold_age if tl_hold_age is not None else -1:.2f},"
                    f"hold_stopline_id={getattr(self, 'tl_hold_stopline_id', None)},"
                    f"active_stopline_id={tl_hold_ctx.get('active_stopline_id') if isinstance(tl_hold_ctx, dict) else None},"
                    f"route_valid={int(bool(tl_hold_ctx.get('route_valid'))) if isinstance(tl_hold_ctx, dict) else 0},"
                    f"decision={decision_name},should_stop={int(bool(should_stop))}"
                    "|ROUTE_AGENT_APPLY_STOP"
                )
            self.reset_longitudinal_memory()
        elif target_speed <= 0.01:
            stage = str(self.get_stage() or "")
            if structured_decision and should_stop:
                control = self.hard_stop_control()
                if current_speed <= 0.20:
                    control.hand_brake = True
                self.reset_longitudinal_memory()
                reason = f"{reason}|structured_should_stop"
            elif stage == "PASSENGER_STOP":
                # Passenger stop: enforce hard stop while in passenger stop stage
                control = self.hard_stop_control()
                if current_speed <= 0.20:
                    control.hand_brake = True
                self.reset_longitudinal_memory()
                reason = f"{reason}|passenger_stop_hard_stop"
            elif self.red_light_hard_stop_enabled and self.is_red_light_stop_reason(reason):
                control = self.hard_stop_control()
                if current_speed <= self.red_light_hard_stop_handbrake_below_mps:
                    control.hand_brake = True
                self.reset_longitudinal_memory()
                reason = f"{reason}|red_light_hard_stop"
            else:
                control = self.smooth_stop_control(reason)
        else:
            ok = self.set_agent_destination_if_needed()
            if not ok:
                control = self.hard_stop_control()
                reason = "route_missing_stop|ROUTE_AGENT_APPLY_STOP"
                target_speed = 0.0
            else:
                reason = f"{reason}|ROUTE_AGENT_APPLY_GO"
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
                red_light_control = (not structured_decision) and "red_light_" in reason_l
                red_light_stop_control = (
                    self.is_red_light_stop_reason(reason)
                    if not structured_decision
                    else bool(should_stop)
                )
                red_light_approach_control = (red_light_control or tl_caution_control) and not red_light_stop_control

                if safety_stopline_control:
                    desired_throttle = 0.0
                    if emergency_brake or (stop_distance_m is not None and stop_distance_m <= 0.50):
                        desired_brake = 1.0
                        reason = f"{reason}|TL_HARD_STOP"
                    elif red_light_stop_control:
                        desired_brake = self.clamp(
                            0.22 + 0.20 * max(0.0, overspeed),
                            0.18,
                            0.85,
                        )
                    elif overspeed > 0.10:
                        desired_brake = self.clamp(0.12 + 0.18 * overspeed, 0.12, 0.65)
                    else:
                        desired_brake = 0.08

                elif speed_error > 0.20:
                    if red_light_approach_control:
                        # Kırmızı yaklaşma modlarında gaz tamamen kesilmez.
                        # far_slow/sensor_slow/crawl ayrı davranır.
                        if "crawl" in reason_l:
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

                    desired_brake = 0.0

                elif (red_light_control or tl_caution_control) and overspeed > 0.10:
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
                    ((not structured_decision) and "red_light_" in str(reason).lower())
                    or safety_stopline_control
                    or tl_caution_control
                )
                brake_slew_step = 1.0 if emergency_brake else (0.22 if safety_stopline_control else (0.060 if red_light_control else 0.018))
                brake_cmd = _slew(
                    self.last_brake_cmd,
                    desired_brake,
                    brake_slew_step,
                )
                if emergency_brake or (safety_stopline_control and stop_distance_m is not None and stop_distance_m <= 0.50):
                    brake_cmd = max(brake_cmd, 0.85)

                if brake_cmd > 0.001:
                    throttle_cmd = 0.0

                self.last_throttle_cmd = throttle_cmd
                self.last_brake_cmd = brake_cmd

                control.throttle = self.clamp(throttle_cmd, 0.0, 0.90)
                red_light_control = "red_light_" in str(reason).lower() or safety_stopline_control or tl_caution_control
                if emergency_brake or (safety_stopline_control and stop_distance_m is not None and stop_distance_m <= 0.50):
                    brake_max = 1.0
                elif safety_stopline_control:
                    brake_max = 0.85
                elif red_light_control:
                    brake_max = 0.45
                else:
                    brake_max = 0.035
                control.brake = self.clamp(
                    brake_cmd,
                    0.0,
                    brake_max,
                )
                control.hand_brake = False
                control.manual_gear_shift = False

        # HARD SAFETY CLAMP: araç kontrolden çıkmasın diye hız/gaz sınırı
        try:
            control.throttle = self.clamp(control.throttle, 0.0, 0.90)
            control.steer = self.clamp(control.steer, -self.max_steer, self.max_steer)
        except Exception:
            pass

        if (
            safety_stopline_control
            or release_confirmed_green
            or "tl_" in str(reason).lower()
            or "traffic_light" in str(stop_reason).lower()
        ) and "CONTROL_TL_FINAL" not in str(reason):
            reason = f"{reason}|CONTROL_TL_FINAL"

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
                time.time() < float(getattr(self, "green_release_force_until", 0.0) or 0.0)
            )

            if (
                float(target_speed) > 0.30
                and float(current_speed) < 0.20
                and ("decision_go" in reason_l or red_approach_bootstrap or green_release_force_active)
                and "timeout" not in reason_l
                and not bool(getattr(self, "tl_stop_hold_active", False))
                and "green_wait_stable" not in reason_l
                and "decision_hold_after_stop" not in reason_l
                and "tl_stop_hold_until_confirmed_green" not in reason_l
                and not safety_stopline_control
                and not should_stop
                and not emergency_brake
            ):
                if red_approach_bootstrap:
                    if "crawl" in reason_l:
                        min_start_throttle = 0.14
                    else:
                        min_start_throttle = 0.26
                    tag = "standstill_red_approach_bootstrap"
                else:
                    min_start_throttle = 0.38
                    tag = "standstill_go_bootstrap"

                control.throttle = max(float(getattr(control, "throttle", 0.0) or 0.0), min_start_throttle)
                control.brake = 0.0
                reason = str(reason) + "|" + tag
                self.last_throttle_cmd = float(control.throttle)
                self.last_brake_cmd = 0.0
        except Exception:
            pass

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
            "decision_should_stop": bool(should_stop),
            "decision_emergency_brake": bool(emergency_brake),
            "decision_stop_reason": latest_decision.get("stop_reason"),
            "decision_stop_distance_m": stop_distance_m,
            "decision_tl_state_machine_state": latest_decision.get("tl_state_machine_state"),
            "decision_stopline_id": latest_decision.get("stopline_id"),
            "decision_stop_target_dist_m": latest_decision.get("stop_target_dist_m"),
            "tl_stop_hold_active": bool(getattr(self, "tl_stop_hold_active", False)),
            "tl_hold_valid": bool(valid_tl_hold),
            "tl_hold_source": getattr(self, "tl_hold_source", None),
            "tl_hold_age_s": round(float(tl_hold_age), 3) if tl_hold_age is not None else None,
            "tl_hold_stopline_id": getattr(self, "tl_hold_stopline_id", None),
            "active_stopline_id": tl_hold_ctx.get("active_stopline_id") if isinstance(tl_hold_ctx, dict) else None,
            "active_stopline_route_valid": bool(tl_hold_ctx.get("route_valid")) if isinstance(tl_hold_ctx, dict) else False,
            "tl_hold_invalid_reason": tl_hold_invalid_reason,
            "release_confirmed_green": bool(release_confirmed_green),
            "route_blocked": bool(route_blocked),
            "traffic_light_state": latest_decision.get("traffic_light_state"),
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
