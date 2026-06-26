import json
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from teknofest_common.runtime_logging import RuntimeJsonlLogger


def _clamp(v, a, b):
    return max(a, min(b, v))


def _safe_float(v):
    if v is None:
        return None
    try:
        return float(v)
    except Exception:
        return None


class ControlNode(Node):
    def __init__(self):
        super().__init__("control_node")

        self.declare_parameter("kp", 0.35)
        self.declare_parameter("ki", 0.02)
        self.declare_parameter("brake_kp", 0.45)
        self.declare_parameter("max_throttle", 0.45)
        self.declare_parameter("max_brake", 0.75)
        self.declare_parameter("steer_rate_limit", 0.08)
        self.declare_parameter("throttle_floor_when_moving", 0.12)
        self.declare_parameter("uphill_speed_error_boost", 0.10)
        self.declare_parameter("min_speed_for_throttle_floor_mps", 0.5)
        self.declare_parameter("throttle_slew_limit", 0.04)
        self.declare_parameter("integral_limit", 3.0)
        self.declare_parameter("command_timeout_s", 0.5)
        self.declare_parameter("red_approach_throttle_cut_distance_m", 8.0)
        self.declare_parameter("red_approach_hard_stop_distance_m", 2.8)
        self.declare_parameter("red_approach_slowdown_target_speed_mps", 0.8)
        self.declare_parameter("red_approach_slowdown_max_throttle", 0.08)
        self.declare_parameter("red_approach_coast_brake_max", 0.0)
        self.declare_parameter("red_approach_no_pid_brake_distance_m", 25.0)
        self.declare_parameter("red_approach_max_throttle", 0.16)
        self.declare_parameter("stop_release_debounce_s", 1.2)
        self.declare_parameter("post_green_ignore_release_block_s", 1.5)
        self.declare_parameter("rate_hz", 20.0)

        self.kp = float(self.get_parameter("kp").value)
        self.ki = float(self.get_parameter("ki").value)
        self.brake_kp = float(self.get_parameter("brake_kp").value)
        self.max_throttle = float(self.get_parameter("max_throttle").value)
        self.max_brake = float(self.get_parameter("max_brake").value)
        self.steer_rate_limit = float(self.get_parameter("steer_rate_limit").value)
        self.throttle_floor_when_moving = float(self.get_parameter("throttle_floor_when_moving").value)
        self.uphill_speed_error_boost = float(self.get_parameter("uphill_speed_error_boost").value)
        self.min_speed_for_throttle_floor_mps = float(self.get_parameter("min_speed_for_throttle_floor_mps").value)
        self.throttle_slew_limit = float(self.get_parameter("throttle_slew_limit").value)
        self.integral_limit = float(self.get_parameter("integral_limit").value)
        self.command_timeout_s = float(self.get_parameter("command_timeout_s").value)
        self.red_approach_throttle_cut_distance_m = float(
            self.get_parameter("red_approach_throttle_cut_distance_m").value
        )
        self.red_approach_hard_stop_distance_m = float(
            self.get_parameter("red_approach_hard_stop_distance_m").value
        )
        self.red_approach_slowdown_target_speed_mps = float(
            self.get_parameter("red_approach_slowdown_target_speed_mps").value
        )
        self.red_approach_slowdown_max_throttle = float(
            self.get_parameter("red_approach_slowdown_max_throttle").value
        )
        self.red_approach_coast_brake_max = float(
            self.get_parameter("red_approach_coast_brake_max").value
        )
        self.red_approach_no_pid_brake_distance_m = float(
            self.get_parameter("red_approach_no_pid_brake_distance_m").value
        )
        self.red_approach_max_throttle = float(
            self.get_parameter("red_approach_max_throttle").value
        )
        self.stop_release_debounce_s = float(self.get_parameter("stop_release_debounce_s").value)
        self.post_green_ignore_release_block_s = float(
            self.get_parameter("post_green_ignore_release_block_s").value
        )
        self.rate_hz = float(self.get_parameter("rate_hz").value)

        self.last_plan = None
        self.last_plan_time = 0.0
        self.last_status = None
        self.last_status_time = 0.0

        self.integral = 0.0
        self.last_throttle = 0.0
        self.last_steer = 0.0
        self.stop_hold_active = False
        self.stop_hold_reason = None
        self.stop_hold_last_stop_time = 0.0
        self.stop_hold_release_candidate_since = None
        self.last_red_approach_guard_time = 0.0
        self._last_logged_event_stop_reason = None
        self.runtime_logger = RuntimeJsonlLogger(
            node_name="control_node",
            file_name="control_node.jsonl",
        )
        self.runtime_logger.update_summary({
            "control_node_log": self.runtime_logger.path(),
        })
        self.get_logger().info(
            f"ControlNode JSONL logging -> {self.runtime_logger.path()}"
        )

        self.create_subscription(String, "/adas/planning/lane_plan", self._plan_cb, 10)
        self.create_subscription(String, "/adas/carla/status", self._status_cb, 10)

        self.cmd_pub = self.create_publisher(String, "/adas/control/vehicle_command", 10)
        self.debug_pub = self.create_publisher(String, "/adas/control/debug", 10)

        self.timer = self.create_timer(1.0 / max(1.0, self.rate_hz), self._run)

    def _plan_cb(self, msg: String):
        try:
            self.last_plan = json.loads(msg.data)
            self.last_plan_time = time.time()
        except Exception:
            self.get_logger().warn("Failed to parse lane_plan JSON")

    def _status_cb(self, msg: String):
        try:
            self.last_status = json.loads(msg.data)
            self.last_status_time = time.time()
        except Exception:
            self.get_logger().warn("Failed to parse carla status JSON")

    def _get_current_speed(self):
        if not self.last_status:
            return 0.0
        s = self.last_status.get("speed_mps")
        if s is not None:
            return float(s)
        s = self.last_status.get("speed")
        if s is not None:
            return float(s)
        s = self.last_status.get("speed_kmh")
        if s is not None:
            return float(s) / 3.6
        return 0.0

    def _run(self):
        now = time.time()
        plan_ok = (self.last_plan is not None) and (now - self.last_plan_time) < 1.0
        status_ok = (self.last_status is not None) and (now - self.last_status_time) < 1.0
        timeout_stop = not plan_ok or not status_ok

        current_speed = self._get_current_speed()
        target_speed = 0.0
        target_speed_before_guard = 0.0
        target_steer = 0.0
        raw_throttle = 0.0
        route_event = "clear"
        route_event_reason = None
        route_event_distance_m = None
        route_event_distance_float = None
        event_stop = False
        event_stop_reason = None
        stop_hold = self.stop_hold_active
        stop_hold_reason = self.stop_hold_reason
        stop_release_candidate = False
        stable_clear_duration_s = 0.0
        last_stop_event_age_s = None
        stop_chatter_guard_active = False
        stop_release_blocked_reason = None
        red_approach_coast_brake_guard_active = False
        red_approach_slowdown_guard_active = False
        red_approach_close_guard_active = False
        post_green_ignore_release_blocked = False
        brake_before_red_approach_coast_guard = None
        brake_after_red_approach_coast_guard = None
        throttle_before_red_approach_limit = None
        throttle_after_red_approach_limit = None

        if plan_ok:
            target_speed = float(self.last_plan.get("target_speed_mps", 0.0))
            target_speed_before_guard = target_speed
            target_steer = float(self.last_plan.get("steer", 0.0))
            route_event = str(self.last_plan.get("route_event", "clear"))
            route_event_reason = self.last_plan.get("route_event_reason")
            route_event_distance_m = self.last_plan.get("route_event_distance_m")
            route_event_distance_float = _safe_float(route_event_distance_m)
            event_stop = route_event in (
                "vehicle_stop",
                "pedestrian_stop",
                "traffic_light_red_stop",
                "traffic_light_yellow_stop",
            )
            if event_stop:
                event_stop_reason = route_event
            if route_event in (
                "traffic_light_red_stop",
                "traffic_light_yellow_stop",
            ):
                target_speed = 0.0
                self.stop_hold_active = True
                self.stop_hold_reason = (
                    str(route_event_reason)
                    if route_event_reason is not None
                    else route_event
                )
                self.stop_hold_last_stop_time = now
                self.stop_hold_release_candidate_since = None
                stop_hold = True
                stop_hold_reason = self.stop_hold_reason

            red_approach_slowdown_guard_active = (
                route_event == "traffic_light_red_approach"
                and route_event_distance_float is not None
                and route_event_distance_float <= self.red_approach_throttle_cut_distance_m
            )
            red_approach_close_guard_active = (
                route_event == "traffic_light_red_approach"
                and route_event_distance_float is not None
                and route_event_distance_float <= self.red_approach_hard_stop_distance_m
            )
            if red_approach_slowdown_guard_active:
                self.last_red_approach_guard_time = now
                slowdown_target = max(0.0, self.red_approach_slowdown_target_speed_mps)
                target_speed = min(
                    target_speed,
                    slowdown_target,
                )
                if not red_approach_close_guard_active and target_speed <= 0.05:
                    target_speed = slowdown_target
            if red_approach_close_guard_active:
                target_speed = 0.0
                event_stop = True
                event_stop_reason = "red_approach_close"
                self.stop_hold_active = True
                self.stop_hold_reason = "red_approach_close"
                self.stop_hold_last_stop_time = now
                self.stop_hold_release_candidate_since = None
                stop_hold = True
                stop_hold_reason = self.stop_hold_reason

            red_approach_coast_brake_guard_active = (
                route_event == "traffic_light_red_approach"
                and route_event_distance_float is not None
                and route_event_distance_float <= self.red_approach_no_pid_brake_distance_m
                and not red_approach_close_guard_active
                and not self.stop_hold_active
            )

        green_release_event = route_event in (
            "traffic_light_green_clear",
            "traffic_light_green_release",
        )
        release_event = route_event in (
            "clear",
            "traffic_light_green_clear",
            "traffic_light_green_release",
        )
        last_stop_event_age_s = (
            now - self.stop_hold_last_stop_time
            if self.stop_hold_last_stop_time > 0.0
            else None
        )
        post_green_same_light_ignore = (
            route_event == "clear"
            and str(route_event_reason or "") == "post_green_same_light_ignore"
        )
        post_green_ignore_release_blocked = (
            post_green_same_light_ignore
            and (
                (
                    last_stop_event_age_s is not None
                    and last_stop_event_age_s < max(0.0, self.post_green_ignore_release_block_s)
                )
                or (
                    self.last_red_approach_guard_time > 0.0
                    and now - self.last_red_approach_guard_time
                    < max(0.0, self.post_green_ignore_release_block_s)
                )
            )
        )
        if post_green_ignore_release_blocked and not self.stop_hold_active:
            target_speed = min(
                target_speed,
                max(0.0, self.red_approach_slowdown_target_speed_mps),
            )
        stop_release_candidate = (
            release_event
            and plan_ok
            and bool(self.last_plan.get("route_ok", False))
            and bool(self.last_plan.get("status_ok", False))
            and target_speed > 0.0
            and (self.stop_hold_active or green_release_event)
        )
        if stop_release_candidate:
            if self.stop_hold_release_candidate_since is None:
                self.stop_hold_release_candidate_since = now
            stable_clear_duration_s = now - self.stop_hold_release_candidate_since
        else:
            self.stop_hold_release_candidate_since = None

        release_stop_hold = (
            stop_release_candidate
            and not post_green_ignore_release_blocked
            and (
                green_release_event
                or stable_clear_duration_s >= max(0.0, self.stop_release_debounce_s)
            )
        )

        if self.stop_hold_active and not release_stop_hold:
            stop_chatter_guard_active = True
            if red_approach_close_guard_active:
                stop_release_blocked_reason = "red_approach_close"
            elif post_green_ignore_release_blocked:
                stop_release_blocked_reason = "recent_red_guard"
            elif stop_release_candidate:
                stop_release_blocked_reason = "unstable_clear"
            elif not release_event:
                stop_release_blocked_reason = "waiting_for_clear"
            elif target_speed <= 0.0:
                stop_release_blocked_reason = "waiting_for_positive_target_speed"
            target_speed = 0.0
            raw_throttle = 0.0
            self.integral = 0.0

        if timeout_stop:
            self.integral = 0.0
            throttle = 0.0
            brake = 1.0
            steer = 0.0
        else:
            error = target_speed - current_speed
            dt = 1.0 / max(1.0, self.rate_hz)
            if target_speed <= 0.05 or error < 0:
                self.integral += error * dt * 0.5
            else:
                self.integral += error * dt

            self.integral = _clamp(self.integral, -self.integral_limit, self.integral_limit)

            raw_throttle = self.kp * error + self.ki * self.integral

            if current_speed > self.min_speed_for_throttle_floor_mps and target_speed > 0.2:
                raw_throttle = max(raw_throttle, self.throttle_floor_when_moving)

            if error > 0.5:
                raw_throttle += self.uphill_speed_error_boost * min(error, 1.0)

            throttle = _clamp(raw_throttle, 0.0, self.max_throttle)
            if (
                route_event == "traffic_light_red_approach"
                and not red_approach_close_guard_active
                and not self.stop_hold_active
            ):
                throttle_before_red_approach_limit = throttle
                throttle = min(
                    throttle,
                    max(0.0, self.red_approach_max_throttle),
                )
                throttle_after_red_approach_limit = throttle
            if red_approach_slowdown_guard_active or post_green_ignore_release_blocked:
                if throttle_before_red_approach_limit is None:
                    throttle_before_red_approach_limit = throttle
                throttle = min(
                    throttle,
                    max(0.0, self.red_approach_slowdown_max_throttle),
                )
                throttle_after_red_approach_limit = throttle
            brake = 0.0

            if error < -0.05:
                brake = _clamp(-self.brake_kp * error, 0.0, self.max_brake)
                throttle = 0.0

            if target_speed <= 0.1 and current_speed > 0.3:
                brake = _clamp(
                    max(0.15, self.brake_kp * current_speed),
                    0.0,
                    self.max_brake,
                )
                if (
                    event_stop
                    and route_event_distance_float is not None
                    and route_event_distance_float <= 1.5
                ):
                    brake = min(max(brake, 0.35), self.max_brake)
                throttle = 0.0
            elif target_speed <= 0.1 and current_speed < 0.2:
                throttle = 0.0
                brake = min(0.2, self.max_brake)
                if not red_approach_slowdown_guard_active or red_approach_close_guard_active:
                    stop_hold = True
                    event_stop = event_stop or stop_hold
                    stop_hold_reason = self.stop_hold_reason
                    if stop_hold_reason is None:
                        stop_hold_reason = (
                            route_event
                            if route_event != "clear"
                            else "zero_target_speed"
                        )
                    if event_stop_reason is None:
                        event_stop_reason = stop_hold_reason
                    if not self.stop_hold_active:
                        self.stop_hold_last_stop_time = now
                        self.stop_hold_release_candidate_since = None
                    self.stop_hold_active = True
                    self.stop_hold_reason = stop_hold_reason
            elif target_speed <= 0.1:
                throttle = 0.0
                brake = min(0.2, self.max_brake)

            brake_before_red_approach_coast_guard = brake
            if red_approach_coast_brake_guard_active and brake > self.red_approach_coast_brake_max:
                brake = min(
                    brake,
                    max(0.0, self.red_approach_coast_brake_max),
                )
            brake_after_red_approach_coast_guard = brake

            if self.stop_hold_active and not release_stop_hold:
                stop_hold = True
                stop_hold_reason = self.stop_hold_reason
                event_stop = True
                if event_stop_reason is None:
                    event_stop_reason = stop_hold_reason
                throttle = 0.0
                brake = min(
                    max(brake, 0.2 if current_speed < 0.2 else 0.35),
                    self.max_brake,
                )

            if release_stop_hold:
                self.stop_hold_active = False
                self.stop_hold_reason = None
                self.stop_hold_release_candidate_since = None
                stop_hold = False
                stop_hold_reason = None
                event_stop = False
                event_stop_reason = None
                brake = 0.0
                if target_speed > current_speed:
                    throttle = _clamp(
                        max(throttle, raw_throttle, 0.01),
                        0.0,
                        self.max_throttle,
                    )

            delta = throttle - self.last_throttle
            delta = _clamp(delta, -self.throttle_slew_limit, self.throttle_slew_limit)
            throttle = _clamp(self.last_throttle + delta, 0.0, 1.0)

            if brake > 0.05:
                throttle = 0.0
            elif throttle > 0.05:
                brake = 0.0

            if brake > 0.05 or target_speed <= 0.05:
                self.integral = 0.0

            steer = _clamp(target_steer, -1.0, 1.0)
            steer_delta = steer - self.last_steer
            steer = self.last_steer + _clamp(steer_delta, -self.steer_rate_limit, self.steer_rate_limit)
            steer = _clamp(steer, -1.0, 1.0)
            self.last_steer = steer

        self.last_throttle = throttle

        cmd = {
            "stamp": now,
            "source": "phase2_control",
            "target_speed_mps": target_speed,
            "effective_target_speed_mps": target_speed,
            "target_speed_before_guard_mps": target_speed_before_guard,
            "current_speed_mps": current_speed,
            "throttle": round(float(throttle), 3),
            "brake": round(float(brake), 3),
            "steer": round(float(self.last_steer), 3),
            "reverse": False,
            "hand_brake": False,
            "stop_hold": stop_hold,
            "event_stop": event_stop,
            "event_stop_reason": event_stop_reason,
            "stop_hold_reason": stop_hold_reason,
            "stop_chatter_guard_active": stop_chatter_guard_active,
            "red_approach_coast_brake_guard_active": red_approach_coast_brake_guard_active,
            "red_approach_slowdown_guard_active": red_approach_slowdown_guard_active,
            "red_approach_close_guard_active": red_approach_close_guard_active,
        }

        m = String()
        m.data = json.dumps(cmd)
        self.cmd_pub.publish(m)

        debug_payload = {
            "target_speed_mps": target_speed,
            "effective_target_speed_mps": target_speed,
            "target_speed_before_guard_mps": target_speed_before_guard,
            "current_speed_mps": current_speed,
            "speed_error": round(target_speed - current_speed, 3),
            "integral": round(self.integral, 3),
            "raw_throttle": round(raw_throttle if not timeout_stop else 0.0, 3),
            "throttle": round(float(throttle), 3),
            "brake": round(float(brake), 3),
            "steer": round(float(self.last_steer), 3),
            "timeout_stop": timeout_stop,
            "stop_hold": stop_hold,
            "event_stop": event_stop,
            "event_stop_reason": event_stop_reason,
            "stop_hold_reason": stop_hold_reason,
            "route_event": route_event,
            "route_event_reason": route_event_reason,
            "route_event_distance_m": route_event_distance_m,
            "stop_chatter_guard_active": stop_chatter_guard_active,
            "stop_release_blocked_reason": stop_release_blocked_reason,
            "stable_clear_duration_s": round(stable_clear_duration_s, 3),
            "last_stop_event_age_s": (
                round(last_stop_event_age_s, 3)
                if last_stop_event_age_s is not None
                else None
            ),
            "stop_release_debounce_s": round(self.stop_release_debounce_s, 3),
            "red_approach_coast_brake_guard_active": red_approach_coast_brake_guard_active,
            "red_approach_coast_brake_max": round(
                self.red_approach_coast_brake_max,
                3,
            ),
            "red_approach_no_pid_brake_distance_m": round(
                self.red_approach_no_pid_brake_distance_m,
                3,
            ),
            "brake_before_red_approach_coast_guard": (
                round(float(brake_before_red_approach_coast_guard), 3)
                if brake_before_red_approach_coast_guard is not None
                else None
            ),
            "brake_after_red_approach_coast_guard": (
                round(float(brake_after_red_approach_coast_guard), 3)
                if brake_after_red_approach_coast_guard is not None
                else None
            ),
            "red_approach_max_throttle": round(self.red_approach_max_throttle, 3),
            "throttle_before_red_approach_limit": (
                round(float(throttle_before_red_approach_limit), 3)
                if throttle_before_red_approach_limit is not None
                else None
            ),
            "throttle_after_red_approach_limit": (
                round(float(throttle_after_red_approach_limit), 3)
                if throttle_after_red_approach_limit is not None
                else None
            ),
            "red_approach_slowdown_guard_active": red_approach_slowdown_guard_active,
            "red_approach_close_guard_active": red_approach_close_guard_active,
            "red_approach_throttle_cut_distance_m": round(
                self.red_approach_throttle_cut_distance_m,
                3,
            ),
            "red_approach_hard_stop_distance_m": round(
                self.red_approach_hard_stop_distance_m,
                3,
            ),
            "red_approach_slowdown_target_speed_mps": round(
                self.red_approach_slowdown_target_speed_mps,
                3,
            ),
            "red_approach_slowdown_max_throttle": round(
                self.red_approach_slowdown_max_throttle,
                3,
            ),
            "post_green_ignore_release_blocked": post_green_ignore_release_blocked,
            "post_green_ignore_release_block_s": round(
                self.post_green_ignore_release_block_s,
                3,
            ),
        }
        dbg = String()
        dbg.data = json.dumps(debug_payload)
        self.debug_pub.publish(dbg)
        if event_stop_reason != self._last_logged_event_stop_reason:
            self._last_logged_event_stop_reason = event_stop_reason
            self.get_logger().info(
                "ControlNode command update: "
                f"route_event={route_event} target_speed={round(target_speed, 3)} "
                f"throttle={cmd['throttle']} brake={cmd['brake']} steer={cmd['steer']} "
                f"event_stop_reason={event_stop_reason}"
            )
        self.runtime_logger.write({
            "kind": "vehicle_command",
            "command": cmd,
            "debug": debug_payload,
            "target_speed_before_guard_mps": round(target_speed_before_guard, 3),
            "target_speed_before_control_mps": round(target_speed, 3),
            "final_vehicle_command": {
                "throttle": cmd["throttle"],
                "brake": cmd["brake"],
                "steer": cmd["steer"],
            },
        })


def main(args=None):
    rclpy.init(args=args)
    node = ControlNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
