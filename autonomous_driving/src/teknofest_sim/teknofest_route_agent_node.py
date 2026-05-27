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
            self.latest_decision = data
            self.last_decision_time = time.time()
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

    def apply_planner_speed_hint_cap(self, target_speed, reason):
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

        capped = min(target_speed, speed_hint)

        if capped < target_speed - 0.01 and "planner_speed_hint" not in str(reason):
            reason = f"{reason}|planner_speed_hint:{speed_hint * 3.6:.1f}kmh/{speed_hint:.2f}mps"

        return capped, reason

    def apply_target_speed_smoothing(self, target_speed, reason):
        target_speed = float(target_speed)

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

        if ("green_light_detected" in str(reason) or "green_light_confirmed_stable" in str(reason)) and prev <= 0.05 and target_speed > 0.10:
            kickoff = min(target_speed, max(float(self.green_release_min_start_speed_mps), target_speed * 0.35))
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

        red_light_control = "red_light_" in str(reason).lower()
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
            if self.red_light_hard_stop_enabled and self.is_red_light_stop_reason(reason):
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
                red_light_control = "red_light_" in reason_l
                red_light_stop_control = self.is_red_light_stop_reason(reason)
                red_light_approach_control = red_light_control and not red_light_stop_control

                if red_light_stop_control:
                    desired_throttle = 0.0
                    desired_brake = self.clamp(0.12 + 0.18 * max(0.0, overspeed), 0.10, 0.60)

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
                red_light_control = "red_light_" in str(reason).lower()
                brake_cmd = _slew(
                    self.last_brake_cmd,
                    desired_brake,
                    0.060 if red_light_control else 0.018,
                )

                if brake_cmd > 0.001:
                    throttle_cmd = 0.0

                self.last_throttle_cmd = throttle_cmd
                self.last_brake_cmd = brake_cmd

                control.throttle = self.clamp(throttle_cmd, 0.0, 0.90)
                red_light_control = "red_light_" in str(reason).lower()
                control.brake = self.clamp(
                    brake_cmd,
                    0.0,
                    0.45 if red_light_control else 0.035,
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

            if (
                float(target_speed) > 0.30
                and float(current_speed) < 0.20
                and ("decision_go" in reason_l or red_approach_bootstrap)
                and "timeout" not in reason_l
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
