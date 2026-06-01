import json
import math
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from teknofest_sim.carla_loader import load_carla


class Phase1BehaviorNode(Node):
    def __init__(self):
        super().__init__("phase1_behavior_node")

        self.declare_parameter("carla_root", "/home/ilker/simulators/CARLA_0.9.15")
        self.declare_parameter("host", "127.0.0.1")
        self.declare_parameter("port", 2000)
        self.declare_parameter("timeout", 120.0)
        self.declare_parameter("ego_role_name", "ego_vehicle")
        self.declare_parameter("route_topic", "/adas/phase1/route")
        self.declare_parameter("lane_command_topic", "/adas/phase1/lane_command")
        self.declare_parameter("perception_events_topic", "/adas/perception/decision_events_json")
        self.declare_parameter("behavior_topic", "/adas/phase1/behavior")
        self.declare_parameter("publish_rate_hz", 20.0)
        self.declare_parameter("route_timeout_s", 1.0)
        self.declare_parameter("lane_timeout_s", 0.6)
        self.declare_parameter("perception_timeout_s", 1.0)
        self.declare_parameter("base_speed_mps", 6.0)
        self.declare_parameter("creep_speed_mps", 0.5)
        self.declare_parameter("red_conf_threshold", 0.50)
        self.declare_parameter("green_conf_threshold", 0.55)
        self.declare_parameter("yellow_conf_threshold", 0.50)
        self.declare_parameter("stopline_search_distance_m", 60.0)
        self.declare_parameter("stop_margin_m", 1.0)
        self.declare_parameter("red_stop_distance_m", 12.0)
        self.declare_parameter("red_no_stopline_speed_cap_mps", 1.5)
        self.declare_parameter("green_confirm_frames", 2)
        self.declare_parameter("post_green_release_ignore_red_s", 2.0)
        self.declare_parameter("post_green_min_speed_mps", 2.0)
        self.declare_parameter("unknown_red_hold_grace_s", 1.0)

        self.carla = load_carla(str(self.get_parameter("carla_root").value))
        self.client = self.carla.Client(
            str(self.get_parameter("host").value),
            int(self.get_parameter("port").value),
        )
        self.client.set_timeout(float(self.get_parameter("timeout").value))
        self.world = self.client.get_world()
        self.map = self.world.get_map()
        self.ego_role_name = str(self.get_parameter("ego_role_name").value)
        self.ego_vehicle = self.wait_for_ego_vehicle()

        self.latest_route = None
        self.latest_route_time = 0.0
        self.latest_lane = None
        self.latest_lane_time = 0.0
        self.latest_tl = {
            "state": "unknown",
            "confidence": 0.0,
            "stamp": 0.0,
        }

        self.red_hold = False
        self.red_hold_since = 0.0
        self.red_hold_last_verified = 0.0
        self.red_hold_stopline_key = None
        self.last_green_seen = 0.0
        self.green_confirm_count = 0
        self.post_green_release_until = 0.0
        self.just_released_green = False

        self.create_subscription(
            String,
            str(self.get_parameter("route_topic").value),
            self.route_callback,
            10,
        )
        self.create_subscription(
            String,
            str(self.get_parameter("lane_command_topic").value),
            self.lane_callback,
            10,
        )
        self.create_subscription(
            String,
            str(self.get_parameter("perception_events_topic").value),
            self.perception_callback,
            10,
        )

        self.pub = self.create_publisher(
            String,
            str(self.get_parameter("behavior_topic").value),
            10,
        )

        rate = float(self.get_parameter("publish_rate_hz").value)
        self.timer = self.create_timer(1.0 / max(rate, 1.0), self.tick)

        self.get_logger().info("PHASE1_BEHAVIOR_READY")

    def wait_for_ego_vehicle(self):
        deadline = time.time() + 30.0
        while time.time() < deadline:
            for vehicle in self.world.get_actors().filter("vehicle.*"):
                if vehicle.attributes.get("role_name", "") == self.ego_role_name:
                    return vehicle
            time.sleep(0.2)
        raise RuntimeError("Phase1 behavior node ego vehicle not found")

    def route_callback(self, msg):
        try:
            self.latest_route = json.loads(msg.data)
            self.latest_route_time = self.as_float(
                self.latest_route.get("stamp"),
                time.time(),
            )
        except Exception as exc:
            self.get_logger().warning(f"route parse error: {exc}")

    def lane_callback(self, msg):
        try:
            self.latest_lane = json.loads(msg.data)
            self.latest_lane_time = self.as_float(
                self.latest_lane.get("stamp"),
                time.time(),
            )
        except Exception as exc:
            self.get_logger().warning(f"lane parse error: {exc}")

    @staticmethod
    def norm_state(value):
        state = str(value or "unknown").strip().lower()
        if state in {"red", "yellow", "green"}:
            return state
        return "unknown"

    def norm_carla_tl_state(self, value):
        text = str(value or "").split(".")[-1].strip().lower()
        aliases = {
            "red": "red",
            "yellow": "yellow",
            "green": "green",
        }
        return aliases.get(text, "unknown")

    @staticmethod
    def as_float(value, default=0.0):
        try:
            if value is None:
                return float(default)
            return float(value)
        except Exception:
            return float(default)

    def perception_callback(self, msg):
        try:
            data = json.loads(msg.data)
        except Exception as exc:
            self.get_logger().warning(f"perception parse error: {exc}")
            return

        active = data.get("active_traffic_light") or {}
        state = self.norm_state(active.get("state"))
        confidence = self.as_float(active.get("state_confidence"), 0.0)

        for event in data.get("events", []) or []:
            if not isinstance(event, dict):
                continue
            if event.get("event_type") != "traffic_light":
                continue
            event_state = self.norm_state(event.get("traffic_light_state"))
            event_conf = self.as_float(event.get("confidence"), 0.0)
            if event_state != "unknown" and event_conf >= confidence:
                state = event_state
                confidence = event_conf

        self.latest_tl = {
            "state": state,
            "confidence": confidence,
            "stamp": time.time(),
            "bbox": active.get("bbox"),
            "source": active.get("source"),
        }

    @staticmethod
    def distance_xy(a, b):
        return math.hypot(float(a.x) - float(b.x), float(a.y) - float(b.y))

    @staticmethod
    def forward_lateral(ego_transform, point):
        yaw = math.radians(float(ego_transform.rotation.yaw))
        dx = float(point.x) - float(ego_transform.location.x)
        dy = float(point.y) - float(ego_transform.location.y)
        forward = math.cos(yaw) * dx + math.sin(yaw) * dy
        lateral = -math.sin(yaw) * dx + math.cos(yaw) * dy
        return forward, lateral

    def get_speed_mps(self):
        velocity = self.ego_vehicle.get_velocity()
        return math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)

    def find_stopline_ahead(self, ego_transform):
        max_distance = float(self.get_parameter("stopline_search_distance_m").value)
        ego_wp = self.map.get_waypoint(
            ego_transform.location,
            project_to_road=True,
            lane_type=self.carla.LaneType.Driving,
        )

        best = None
        best_forward = max_distance + 1.0

        for tl in self.world.get_actors().filter("traffic.traffic_light*"):
            try:
                stop_waypoints = tl.get_stop_waypoints()
            except Exception:
                stop_waypoints = []

            for wp in stop_waypoints:
                if wp is None:
                    continue

                location = wp.transform.location
                forward, lateral = self.forward_lateral(ego_transform, location)
                if forward <= 0.0 or forward > max_distance:
                    continue
                if abs(lateral) > 5.0:
                    continue

                same_lane = False
                if ego_wp is not None:
                    same_lane = (
                        int(wp.road_id) == int(ego_wp.road_id)
                        and int(wp.lane_id) == int(ego_wp.lane_id)
                    )

                score = forward + (0.0 if same_lane else 8.0) + abs(lateral) * 1.5
                if score < best_forward:
                    carla_state = "unknown"
                    try:
                        carla_state = self.norm_carla_tl_state(tl.get_state())
                    except Exception:
                        try:
                            carla_state = self.norm_carla_tl_state(tl.state)
                        except Exception:
                            carla_state = "unknown"

                    best_forward = score
                    best = {
                        "distance_m": forward,
                        "lateral_m": lateral,
                        "traffic_light_id": int(tl.id),
                        "road_id": int(wp.road_id),
                        "lane_id": int(wp.lane_id),
                        "same_lane": same_lane,
                        "carla_state": carla_state,
                        "stopline_key": f"{int(wp.road_id)}:{int(wp.lane_id)}",
                    }

        return best

    @staticmethod
    def apply_cap(speed, cap):
        return min(float(speed), float(cap))

    def apply_lane_safety_caps(self, speed, lane_valid, lateral_offset, steering_target):
        reason_parts = []
        unstable_stop = False

        if not lane_valid:
            speed = self.apply_cap(speed, 1.0)
            reason_parts.append("lane_command_missing")

        offset = abs(self.as_float(lateral_offset, 0.0))
        steer = abs(self.as_float(steering_target, 0.0))

        if offset > 2.00:
            speed = 0.0
            unstable_stop = True
            reason_parts.append("lane_unstable_stop")
        elif offset > 1.50:
            speed = self.apply_cap(speed, float(self.get_parameter("creep_speed_mps").value))
            unstable_stop = True
            reason_parts.append("lane_unstable_stop")
        elif offset > 1.00:
            speed = self.apply_cap(speed, 1.2)
            reason_parts.append("lane_offset_speed_cap_gt_1_00")
        elif offset > 0.50:
            speed = self.apply_cap(speed, 3.0)
            reason_parts.append("lane_offset_speed_cap_gt_0_50")

        if steer > 0.45:
            speed = self.apply_cap(speed, 2.0)
            reason_parts.append("lane_offset_speed_cap_steer_gt_0_45")

        return speed, reason_parts, unstable_stop

    def traffic_light_decision(self, now, stopline, current_speed):
        perception_state = self.latest_tl.get("state", "unknown")
        perception_confidence = self.as_float(self.latest_tl.get("confidence"), 0.0)
        age = now - self.as_float(self.latest_tl.get("stamp"), 0.0)
        if age > float(self.get_parameter("perception_timeout_s").value):
            perception_state = "unknown"
            perception_confidence = 0.0

        stopline_distance = None if stopline is None else float(stopline["distance_m"])
        stop_margin = float(self.get_parameter("stop_margin_m").value)

        if stopline_distance is None:
            red_conf = float(self.get_parameter("red_conf_threshold").value)
            if perception_state == "red" and perception_confidence >= red_conf:
                return (
                    "red_seen_no_reliable_stopline",
                    perception_state,
                    perception_confidence,
                    None,
                    True,
                    float(self.get_parameter("red_no_stopline_speed_cap_mps").value),
                )

            self.red_hold = False
            self.red_hold_since = 0.0
            self.red_hold_stopline_key = None
            self.green_confirm_count = 0
            return "no_relevant_stopline", perception_state, perception_confidence, None, False, None

        carla_state = self.norm_state(stopline.get("carla_state"))
        stopline_key = str(stopline.get("stopline_key", "unknown"))

        if carla_state != "unknown":
            state = carla_state
            confidence = 1.0
            source = "carla"
        else:
            state = perception_state
            confidence = perception_confidence
            source = "perception"

        if self.red_hold and self.red_hold_stopline_key not in (None, stopline_key):
            self.red_hold = False
            self.red_hold_since = 0.0
            self.red_hold_stopline_key = None
            self.green_confirm_count = 0

        if (
            state == "green"
            and confidence >= float(self.get_parameter("green_conf_threshold").value)
        ):
            self.green_confirm_count += 1
            self.last_green_seen = now
            green_confirmed = (
                source == "carla"
                or self.green_confirm_count >= int(self.get_parameter("green_confirm_frames").value)
            )
            if self.red_hold and green_confirmed:
                self.red_hold = False
                self.red_hold_since = 0.0
                self.red_hold_stopline_key = None
                self.post_green_release_until = (
                    now
                    + float(self.get_parameter("post_green_release_ignore_red_s").value)
                )
                self.just_released_green = True
        elif state != "unknown":
            self.green_confirm_count = 0

        if stopline_distance < -1.0:
            self.red_hold = False
            self.red_hold_stopline_key = None

        ignore_red_after_green = (
            now <= self.post_green_release_until
            and stopline_key == str(stopline.get("stopline_key", "unknown"))
        )

        if (
            state == "red"
            and confidence >= float(self.get_parameter("red_conf_threshold").value)
            and not ignore_red_after_green
        ):
            self.red_hold = True
            self.red_hold_stopline_key = stopline_key
            self.red_hold_last_verified = now
            if self.red_hold_since <= 0.0:
                self.red_hold_since = now

        if (
            state == "yellow"
            and confidence >= float(self.get_parameter("yellow_conf_threshold").value)
        ):
            stopping_distance = current_speed * current_speed / (2.0 * 3.0) + stop_margin + 1.0
            if stopline_distance > stopping_distance:
                self.red_hold = True
                self.red_hold_stopline_key = stopline_key
                self.red_hold_last_verified = now
                if self.red_hold_since <= 0.0:
                    self.red_hold_since = now

        if self.red_hold and state == "unknown":
            grace = float(self.get_parameter("unknown_red_hold_grace_s").value)
            if now - self.red_hold_last_verified > grace:
                self.red_hold = False
                self.red_hold_since = 0.0
                self.red_hold_stopline_key = None

        if not self.red_hold:
            self.red_hold_since = 0.0
            return "clear", state, confidence, stopline_distance, False, None

        if stopline_distance <= stop_margin + 0.8:
            return "stop_at_red_stopline", state, confidence, stopline_distance, True, 0.0

        red_stop_distance = float(self.get_parameter("red_stop_distance_m").value)
        if stopline_distance <= red_stop_distance:
            return "stop_for_red_stopline", state, confidence, stopline_distance, True, 0.0

        approach_speed = max(
            0.5,
            min(2.5, (stopline_distance - stop_margin) * 0.25),
        )
        return "approach_red_stopline", state, confidence, stopline_distance, True, approach_speed

    def publish_behavior(self, payload):
        msg = String()
        msg.data = json.dumps(payload)
        self.pub.publish(msg)

    def tick(self):
        now = time.time()
        self.just_released_green = False
        route_age = now - self.latest_route_time
        lane_age = now - self.latest_lane_time

        route_valid = (
            self.latest_route is not None
            and route_age <= float(self.get_parameter("route_timeout_s").value)
            and bool(self.latest_route.get("valid", False))
        )
        route_reason = "route_missing"
        if self.latest_route is not None:
            route_reason = str(self.latest_route.get("reason", "unknown"))

        lane_valid = (
            self.latest_lane is not None
            and lane_age <= float(self.get_parameter("lane_timeout_s").value)
            and bool(self.latest_lane.get("valid", False))
        )

        steering_target = 0.0
        lateral_offset = None
        lane_confidence = 0.0
        lane_reason = "lane_missing"
        if self.latest_lane is not None:
            steering_target = self.as_float(self.latest_lane.get("steering_target"), 0.0)
            lateral_offset = self.latest_lane.get("lateral_offset_m")
            lane_confidence = self.as_float(self.latest_lane.get("confidence"), 0.0)
            lane_reason = str(self.latest_lane.get("reason", "unknown"))
            if lateral_offset is None:
                lane_valid = False

        current_speed = self.get_speed_mps()
        ego_transform = self.ego_vehicle.get_transform()
        stopline = self.find_stopline_ahead(ego_transform)
        tl_reason, tl_state, tl_conf, stopline_distance, stop_required, tl_speed_cap = (
            self.traffic_light_decision(now, stopline, current_speed)
        )

        decision = "GO"
        reason_parts = []
        target_speed = float(self.get_parameter("base_speed_mps").value)

        if not route_valid:
            decision = "STOP"
            target_speed = 0.0
            reason_parts.append("route_invalid")
            if route_reason not in {"ok", "unknown"}:
                reason_parts.append(route_reason)
        else:
            target_speed, lane_reasons, lane_unstable_stop = self.apply_lane_safety_caps(
                target_speed,
                lane_valid,
                lateral_offset,
                steering_target,
            )
            reason_parts.extend(lane_reasons)
            if lane_unstable_stop:
                decision = "STOP"
                target_speed = 0.0
            elif lane_reasons:
                decision = "SLOW"

        if stop_required and route_valid:
            if tl_speed_cap is None or tl_speed_cap <= 0.0:
                decision = "STOP"
                target_speed = 0.0
            else:
                decision = "SLOW"
                target_speed = min(target_speed, float(tl_speed_cap))
            reason_parts.append(tl_reason)

        if lane_valid and decision != "STOP" and target_speed <= 1.0:
            decision = "SLOW"

        if (
            not lane_valid
            and route_valid
            and lane_reason in {"route_uncertain", "route_lateral_jump"}
        ):
            decision = "STOP"
            target_speed = 0.0

        if (
            self.just_released_green
            and route_valid
            and lane_valid
            and decision != "STOP"
            and not any("lane_offset_speed_cap" in part for part in reason_parts)
        ):
            target_speed = max(
                target_speed,
                float(self.get_parameter("post_green_min_speed_mps").value),
            )
            reason_parts.append("green_release_min_speed")

        payload = {
            "stamp": now,
            "decision": decision,
            "target_speed": round(float(target_speed), 3),
            "steering_target": round(float(steering_target), 4),
            "stop_required": bool(stop_required or decision == "STOP"),
            "reason": "+".join(reason_parts) if reason_parts else "ok",
            "traffic_light_state": tl_state,
            "traffic_light_confidence": round(float(tl_conf), 3),
            "stopline_distance_m": (
                round(float(stopline_distance), 3)
                if stopline_distance is not None
                else None
            ),
            "route_valid": bool(route_valid),
            "route_age": round(route_age, 3) if self.latest_route is not None else None,
            "lane_valid": bool(lane_valid),
            "lane_age": round(lane_age, 3) if self.latest_lane is not None else None,
            "lane_confidence": round(float(lane_confidence), 3),
            "lateral_offset_m": lateral_offset,
            "lane_reason": lane_reason,
            "red_hold": bool(self.red_hold),
            "post_green_release_active": bool(now <= self.post_green_release_until),
            "current_speed": round(float(current_speed), 3),
        }

        if stopline is not None:
            payload["matched_stopline"] = stopline

        self.publish_behavior(payload)

        self.get_logger().info(
            "PHASE1_BEHAVIOR "
            f"decision={decision} target={target_speed:.2f} "
            f"steer={steering_target:.2f} tl={tl_state}:{tl_conf:.2f} "
            f"stopline={payload['stopline_distance_m']} reason={payload['reason']}",
            throttle_duration_sec=0.5,
        )


def main(args=None):
    rclpy.init(args=args)
    node = Phase1BehaviorNode()
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
