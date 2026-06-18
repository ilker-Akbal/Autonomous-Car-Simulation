#!/usr/bin/env python3
from __future__ import annotations

import json
import time
from typing import Optional

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from teknofest_common.runtime_logging import RuntimeJsonlLogger
from teknofest_sim.carla_loader import load_carla


class CarlaControlAdapterNode(Node):
    def __init__(self):
        super().__init__("carla_control_adapter_node")

        # -------------------------
        # Config / parameter block
        # -------------------------
        self.declare_parameter("carla_root", "/home/ilker/simulators/CARLA_0.9.15")
        self.declare_parameter("host", "127.0.0.1")
        self.declare_parameter("port", 2000)
        self.declare_parameter("timeout", 120.0)
        self.declare_parameter("ego_role_name", "ego_vehicle")
        self.declare_parameter("vehicle_command_topic", "/adas/control/vehicle_command")
        self.declare_parameter("adapter_status_topic", "/adas/control/adapter_status")
        self.declare_parameter("apply_period_s", 0.05)
        self.declare_parameter("command_hold_s", 0.7)
        self.declare_parameter("command_timeout_s", 3.0)
        self.declare_parameter("emergency_brake", 0.75)
        self.declare_parameter("log_root", "autonomous_driving/outputs/teknofest_sim_logs")
        self.declare_parameter("log_session_id", "")
        self.declare_parameter("jsonl_logging_enabled", True)
        self.declare_parameter("ros_log_period_s", 1.0)

        self.carla_root = str(self.get_parameter("carla_root").value)
        self.host = str(self.get_parameter("host").value)
        self.port = int(self.get_parameter("port").value)
        self.timeout = float(self.get_parameter("timeout").value)
        self.ego_role_name = str(self.get_parameter("ego_role_name").value)
        self.apply_period_s = float(self.get_parameter("apply_period_s").value)
        self.command_hold_s = float(self.get_parameter("command_hold_s").value)
        self.command_timeout_s = float(self.get_parameter("command_timeout_s").value)
        self.emergency_brake = float(self.get_parameter("emergency_brake").value)
        self.ros_log_period_s = float(self.get_parameter("ros_log_period_s").value)

        # -------------------------
        # Runtime state block
        # -------------------------
        self.carla = None
        self.client = None
        self.world = None
        self.ego_vehicle = None
        self.last_ego_lookup_s = 0.0
        self.last_command_s = 0.0
        self.command_payload: Optional[dict] = None
        self.last_ros_log_s = 0.0

        self.runtime_logger = RuntimeJsonlLogger(
            node_name="carla_control_adapter_node",
            file_name="adapter.jsonl",
            log_root=str(self.get_parameter("log_root").value),
            session_id=str(self.get_parameter("log_session_id").value) or None,
            enabled=bool(self.get_parameter("jsonl_logging_enabled").value),
        )

        # -------------------------
        # Publisher block
        # -------------------------
        self.status_pub = self.create_publisher(
            String,
            str(self.get_parameter("adapter_status_topic").value),
            10,
        )

        # -------------------------
        # Subscriber block
        # -------------------------
        self.create_subscription(
            String,
            str(self.get_parameter("vehicle_command_topic").value),
            self.command_cb,
            10,
        )

        # -------------------------
        # Timer / startup block
        # -------------------------
        self.connect_to_carla()
        self.create_timer(self.apply_period_s, self.tick)
        self.get_logger().info("CARLA control adapter node ready.")

    # -------------------------
    # CARLA helper functions
    # -------------------------
    def connect_to_carla(self):
        self.carla = load_carla(self.carla_root)
        self.client = self.carla.Client(self.host, self.port)
        self.client.set_timeout(self.timeout)
        self.world = self.client.get_world()

    def find_ego_vehicle(self):
        now = time.time()

        if self.ego_vehicle is not None:
            try:
                if self.ego_vehicle.is_alive:
                    return self.ego_vehicle
            except Exception:
                self.ego_vehicle = None

        if now - self.last_ego_lookup_s < 1.0:
            return self.ego_vehicle

        self.last_ego_lookup_s = now
        vehicles = self.world.get_actors().filter("vehicle.*")

        for vehicle in vehicles:
            if vehicle.attributes.get("role_name", "") == self.ego_role_name:
                self.ego_vehicle = vehicle
                self.get_logger().info(f"Ego vehicle found for control adapter: id={vehicle.id}")
                return vehicle

        return None

    # -------------------------
    # Subscriber callbacks
    # -------------------------
    def command_cb(self, msg: String):
        try:
            self.command_payload = json.loads(msg.data)
            self.last_command_s = time.time()
        except json.JSONDecodeError as exc:
            self.get_logger().warning(f"Invalid control command JSON ignored: {exc}")

    # -------------------------
    # Control adapter functions
    # -------------------------
    def clamp(self, value: float, low: float, high: float) -> float:
        return max(low, min(high, value))

    def build_vehicle_control(self, payload: Optional[dict], command_age_s: Optional[float]):
        if not payload or command_age_s is None or command_age_s > self.command_timeout_s:
            return self.carla.VehicleControl(
                throttle=0.0,
                steer=0.0,
                brake=self.clamp(self.emergency_brake, 0.0, 1.0),
                hand_brake=False,
                reverse=False,
            ), "command_timeout"

        if command_age_s > self.command_hold_s:
            return self.carla.VehicleControl(
                throttle=0.0,
                steer=self.clamp(float(payload.get("steer", 0.0)), -1.0, 1.0),
                brake=0.0,
                hand_brake=False,
                reverse=bool(payload.get("reverse", False)),
            ), "command_coast_timeout"

        throttle = self.clamp(float(payload.get("throttle", 0.0)), 0.0, 1.0)
        brake = self.clamp(float(payload.get("brake", 0.0)), 0.0, 1.0)
        reason = str(payload.get("reason", "command"))
        red_reason_text = " ".join((
            reason,
            str(payload.get("stop_reason") or ""),
            str(payload.get("tl_decision_reason") or ""),
        )).lower()
        tl_state_machine_state = str(payload.get("tl_state_machine_state") or "")
        associated_light_color = str(payload.get("associated_light_color") or "").lower()
        current_speed = float(payload.get("current_speed_mps", 0.0) or 0.0)
        target_speed = float(payload.get("target_speed_mps", 0.0) or 0.0)
        visual_red_approach = bool(payload.get("visual_red_approach", False))
        visual_red_commit = bool(payload.get("visual_red_commit", False))
        visual_red_brake = bool(payload.get("visual_red_brake", False))
        associated_light_above_stopline = bool(payload.get("associated_light_above_stopline", False))
        try:
            visual_dist_filtered = (
                float(payload.get("visual_dist_filtered"))
                if payload.get("visual_dist_filtered") is not None
                else None
            )
        except (TypeError, ValueError):
            visual_dist_filtered = None
        brake_released_for_visual_approach = bool(
            payload.get("brake_released_for_visual_approach", False)
        )
        red_visual_slow_reason_active = "red_visual_slow_10m_rule" in red_reason_text
        green_release_active = bool(
            reason == "green_release"
            or str(payload.get("tl_decision_reason") or "") == "green_release"
            or tl_state_machine_state == "GREEN_RELEASE"
            or associated_light_color == "green"
        )
        if green_release_active:
            payload["stop_request"] = False
            payload["red_decel_brake_override"] = False
            payload["downhill_red_brake_guard"] = False
            payload["red_decel_brake_override_reason"] = ""
            payload["red_visual_slow_active_brake"] = False
            payload["red_visual_slow_active_brake_reason"] = ""
            payload["red_visual_slow_brake_value"] = 0.0
            payload["red_stop_hard_brake"] = False
            payload["red_stop_hard_brake_reason"] = ""
            payload["red_stop_hard_brake_value"] = 0.0
            payload["red_stop_2m_full_brake"] = False
            payload["red_stop_2m_full_brake_value"] = 0.0
            payload["green_release_brake_cleared"] = True
            payload["brake_release_reason"] = "green_release"
            return self.carla.VehicleControl(
                throttle=throttle,
                steer=self.clamp(float(payload.get("steer", 0.0)), -1.0, 1.0),
                brake=0.0,
                hand_brake=bool(payload.get("hand_brake", False)),
                reverse=bool(payload.get("reverse", False)),
            ), reason

        red_visual_control_active = bool(
            red_visual_slow_reason_active
            or visual_red_approach
            or visual_red_brake
        )
        visual_near_light_window = bool(visual_dist_filtered is not None and visual_dist_filtered <= 6.5)
        red_decel_guard = bool(
            (
                reason in {"red_decel", "visual_red_approach_to_stopline", "red_visual_slow_10m_rule"}
                or red_visual_control_active
            )
            and visual_red_brake
            and (visual_red_commit or not visual_red_approach)
        )
        visual_approach_release = bool(
            visual_red_approach
            and not visual_red_commit
            and not payload.get("stop_request", False)
            and current_speed <= target_speed + 0.25
        )
        if visual_approach_release or brake_released_for_visual_approach:
            brake = 0.0
        if red_decel_guard and not visual_approach_release:
            throttle = 0.0
            brake = max(brake, 0.35)
        elif (
            (
                reason in {"red_decel", "visual_red_approach_to_stopline", "red_visual_slow_10m_rule"}
                or red_visual_control_active
            )
            and payload.get("red_decel_brake_override", False)
            and not visual_approach_release
        ):
            throttle = 0.0
            brake = max(brake, float(payload.get("red_visual_slow_brake_value") or 0.25))
        elif (
            (
                reason in {"red_decel", "visual_red_approach_to_stopline", "red_visual_slow_10m_rule"}
                or red_visual_control_active
            )
            and current_speed > target_speed + 0.25
            and not visual_approach_release
        ):
            throttle = 0.0
            requested_brake = (
                0.60
                if payload.get("downhill_red_brake_guard", False)
                or associated_light_above_stopline
                or visual_near_light_window
                else 0.45
            )
            brake = max(brake, requested_brake)
            payload["red_visual_slow_active_brake"] = True
            payload["red_visual_slow_active_brake_reason"] = "red_visual_slow_active_brake"
            payload["red_visual_slow_brake_value"] = round(float(requested_brake), 3)
            payload["red_decel_brake_override"] = True
            payload["red_decel_brake_override_reason"] = "red_visual_slow_active_brake"
        red_stop_guard = bool(payload.get("stop_request", False) and "red" in red_reason_text)
        if red_stop_guard:
            throttle = 0.0
            red_stop_full_brake_request = bool(
                "red_stop_2m_visual_stopline" in red_reason_text
                or "red_stop_2m_full_brake" in red_reason_text
                or "red_visual_near_light_failsafe_stop" in red_reason_text
                or "locked_red_stop_point_overrun" in red_reason_text
                or tl_state_machine_state == "RED_STOP_COMMIT"
                or "red_stop_commit" in red_reason_text
            )
            requested_stop_brake = 1.0 if red_stop_full_brake_request else 1.0 if current_speed > 0.2 else 0.65
            brake = max(brake, requested_stop_brake)
            if red_stop_full_brake_request:
                payload["red_stop_2m_full_brake"] = True
                payload["red_stop_2m_full_brake_value"] = 1.0
            if current_speed > 0.2 or red_stop_full_brake_request:
                payload["red_stop_hard_brake"] = True
                payload["red_stop_hard_brake_reason"] = (
                    "red_stop_2m_full_brake"
                    if red_stop_full_brake_request
                    else "red_visual_near_light_failsafe_stop"
                    if "red_visual_near_light_failsafe_stop" in red_reason_text
                    else "red_stop_2m_visual_stopline"
                    if "red_stop_2m_visual_stopline" in red_reason_text
                    else "locked_red_stop_point_overrun"
                    if "locked_red_stop_point_overrun" in red_reason_text
                    else "moving_red_stop_request"
                )
                payload["red_stop_hard_brake_value"] = round(float(requested_stop_brake), 3)

        return self.carla.VehicleControl(
            throttle=throttle,
            steer=self.clamp(float(payload.get("steer", 0.0)), -1.0, 1.0),
            brake=self.clamp(brake, 0.0, 1.0),
            hand_brake=bool(payload.get("hand_brake", False)),
            reverse=bool(payload.get("reverse", False)),
        ), reason

    # -------------------------
    # Main timer
    # -------------------------
    def tick(self):
        ego = self.find_ego_vehicle()
        if ego is None:
            self.publish_status(applied=False, reason="missing_ego")
            return

        now = time.time()
        command_age_s = now - self.last_command_s if self.last_command_s > 0.0 else None
        control, control_reason = self.build_vehicle_control(self.command_payload, command_age_s)

        try:
            ego.apply_control(control)
        except Exception as exc:
            self.publish_status(applied=False, reason=f"apply_failed:{exc}")
            return

        self.publish_status(
            applied=True,
            reason=control_reason,
            control=control,
            command_age_s=command_age_s,
        )

    # -------------------------
    # Debug / publish block
    # -------------------------
    def publish_status(
        self,
        *,
        applied: bool,
        reason: str,
        control=None,
        command_age_s: Optional[float] = None,
    ):
        payload = {
            "stamp": time.time(),
            "source": "carla_control_adapter_node",
            "applied": bool(applied),
            "reason": reason,
            "ego_x": (self.command_payload or {}).get("ego_x"),
            "ego_y": (self.command_payload or {}).get("ego_y"),
            "ego_yaw": (self.command_payload or {}).get("ego_yaw"),
            "current_speed_mps": (self.command_payload or {}).get("current_speed_mps"),
            "active_mission_target": (self.command_payload or {}).get("active_mission_target"),
            "mission_state": (self.command_payload or {}).get("mission_state"),
            "route_index": (self.command_payload or {}).get("route_index"),
            "route_length": (self.command_payload or {}).get("route_length"),
            "lane_target_x": (self.command_payload or {}).get("lane_target_x"),
            "lane_target_y": (self.command_payload or {}).get("lane_target_y"),
            "cross_track_error": (self.command_payload or {}).get("cross_track_error"),
            "heading_error": (self.command_payload or {}).get("heading_error"),
            "target_speed_mps": (self.command_payload or {}).get("target_speed_mps"),
            "stop_request": (self.command_payload or {}).get("stop_request"),
            "stop_reason": (self.command_payload or {}).get("stop_reason"),
            "tl_decision_reason": (self.command_payload or {}).get("tl_decision_reason"),
            "red_decel_brake_override": (self.command_payload or {}).get("red_decel_brake_override"),
            "downhill_red_brake_guard": (self.command_payload or {}).get("downhill_red_brake_guard"),
            "red_decel_brake_override_reason": (self.command_payload or {}).get("red_decel_brake_override_reason"),
            "brake_released_for_visual_approach": (self.command_payload or {}).get("brake_released_for_visual_approach"),
            "red_visual_slow_active_brake": (self.command_payload or {}).get("red_visual_slow_active_brake"),
            "red_visual_slow_active_brake_reason": (self.command_payload or {}).get(
                "red_visual_slow_active_brake_reason"
            ),
            "red_visual_slow_brake_value": (self.command_payload or {}).get("red_visual_slow_brake_value"),
            "red_stop_hard_brake": (self.command_payload or {}).get("red_stop_hard_brake"),
            "red_stop_hard_brake_reason": (self.command_payload or {}).get("red_stop_hard_brake_reason"),
            "red_stop_hard_brake_value": (self.command_payload or {}).get("red_stop_hard_brake_value"),
            "red_stop_2m_full_brake": (self.command_payload or {}).get("red_stop_2m_full_brake"),
            "red_stop_2m_full_brake_value": (self.command_payload or {}).get("red_stop_2m_full_brake_value"),
            "green_release_brake_cleared": (self.command_payload or {}).get("green_release_brake_cleared"),
            "brake_release_reason": (self.command_payload or {}).get("brake_release_reason"),
            "visual_red_brake": (self.command_payload or {}).get("visual_red_brake"),
            "visual_red_commit": (self.command_payload or {}).get("visual_red_commit"),
            "visual_red_approach": (self.command_payload or {}).get("visual_red_approach"),
            "associated_light_above_stopline": (self.command_payload or {}).get("associated_light_above_stopline"),
            "associated_light_color": (self.command_payload or {}).get("associated_light_color"),
            "visual_red_approach_target_mps": (self.command_payload or {}).get("visual_red_approach_target_mps"),
            "visual_dist_filtered": (self.command_payload or {}).get("visual_dist_filtered"),
            "tl_state_machine_state": (self.command_payload or {}).get("tl_state_machine_state"),
            "startup_phase": (self.command_payload or {}).get("startup_phase"),
            "throttle_startup_phase": (self.command_payload or {}).get("throttle_startup_phase"),
            "lane_transition_detected": (self.command_payload or {}).get("lane_transition_detected"),
            "route_target_jump_distance": (self.command_payload or {}).get("route_target_jump_distance"),
            "is_junction": (self.command_payload or {}).get("is_junction"),
            "upcoming_turn_type": (self.command_payload or {}).get("upcoming_turn_type"),
            "upcoming_turn_distance_m": (self.command_payload or {}).get("upcoming_turn_distance_m"),
            "turn_slowdown_active": (self.command_payload or {}).get("turn_slowdown_active"),
            "turn_speed_limit_mps": (self.command_payload or {}).get("turn_speed_limit_mps"),
            "turn_state": (self.command_payload or {}).get("turn_state"),
            "speed_state": (self.command_payload or {}).get("speed_state"),
            "speed_setpoint_raw": (self.command_payload or {}).get("speed_setpoint_raw"),
            "speed_setpoint_smoothed": (self.command_payload or {}).get("speed_setpoint_smoothed"),
            "cruise_allowed": (self.command_payload or {}).get("cruise_allowed"),
            "cte_recovery_active": (self.command_payload or {}).get("cte_recovery_active"),
            "junction_offroute_safety_stop": (self.command_payload or {}).get("junction_offroute_safety_stop"),
            "off_route": (self.command_payload or {}).get("off_route"),
            "safety_reason": (self.command_payload or {}).get("safety_reason"),
            "steer_raw": (self.command_payload or {}).get("steer_raw"),
            "steer_smoothed": (self.command_payload or {}).get("steer_smoothed"),
            "steer_rate_limited_value": (self.command_payload or {}).get("steer_rate_limited_value"),
            "steer_rate_limited": (self.command_payload or {}).get("steer_rate_limited"),
            "steer_cap_reason": (self.command_payload or {}).get("steer_cap_reason"),
            "lane_plan_age_ms": (self.command_payload or {}).get("lane_plan_age_ms"),
            "mission_status_age_ms": (self.command_payload or {}).get("mission_status_age_ms"),
            "route_age_ms": (self.command_payload or {}).get("route_age_ms"),
            "ego_status_age_ms": (self.command_payload or {}).get("ego_status_age_ms"),
            "control_cmd_age_ms": int(command_age_s * 1000.0)
            if command_age_s is not None else None,
            "timeout_source": reason if "timeout" in str(reason) else "",
            "timeout_threshold_ms": {
                "command_hold": int(self.command_hold_s * 1000.0),
                "command_timeout": int(self.command_timeout_s * 1000.0),
            },
        }

        if control is not None:
            payload.update(
                {
                    "throttle": round(float(control.throttle), 4),
                    "brake": round(float(control.brake), 4),
                    "steer": round(float(control.steer), 4),
                    "reverse": bool(control.reverse),
                    "hand_brake": bool(control.hand_brake),
                }
            )

        msg = String()
        msg.data = json.dumps(payload, ensure_ascii=False)
        self.status_pub.publish(msg)
        self.log_runtime(payload)

    def log_runtime(self, payload: dict):
        record = {
            "ego_x": payload.get("ego_x"),
            "ego_y": payload.get("ego_y"),
            "ego_yaw": payload.get("ego_yaw"),
            "current_speed_mps": payload.get("current_speed_mps"),
            "active_mission_target": payload.get("active_mission_target"),
            "mission_state": payload.get("mission_state"),
            "route_index": payload.get("route_index"),
            "route_length": payload.get("route_length"),
            "lane_target_x": payload.get("lane_target_x"),
            "lane_target_y": payload.get("lane_target_y"),
            "cross_track_error": payload.get("cross_track_error"),
            "heading_error": payload.get("heading_error"),
            "target_speed_mps": payload.get("target_speed_mps"),
            "stop_request": payload.get("stop_request"),
            "stop_reason": payload.get("stop_reason"),
            "tl_decision_reason": payload.get("tl_decision_reason"),
            "red_decel_brake_override": payload.get("red_decel_brake_override"),
            "downhill_red_brake_guard": payload.get("downhill_red_brake_guard"),
            "red_decel_brake_override_reason": payload.get("red_decel_brake_override_reason"),
            "brake_released_for_visual_approach": payload.get("brake_released_for_visual_approach"),
            "red_visual_slow_active_brake": payload.get("red_visual_slow_active_brake"),
            "red_visual_slow_active_brake_reason": payload.get("red_visual_slow_active_brake_reason"),
            "red_visual_slow_brake_value": payload.get("red_visual_slow_brake_value"),
            "red_stop_hard_brake": payload.get("red_stop_hard_brake"),
            "red_stop_hard_brake_reason": payload.get("red_stop_hard_brake_reason"),
            "red_stop_hard_brake_value": payload.get("red_stop_hard_brake_value"),
            "red_stop_2m_full_brake": payload.get("red_stop_2m_full_brake"),
            "red_stop_2m_full_brake_value": payload.get("red_stop_2m_full_brake_value"),
            "green_release_brake_cleared": payload.get("green_release_brake_cleared"),
            "brake_release_reason": payload.get("brake_release_reason"),
            "visual_red_brake": payload.get("visual_red_brake"),
            "visual_red_commit": payload.get("visual_red_commit"),
            "visual_red_approach": payload.get("visual_red_approach"),
            "associated_light_above_stopline": payload.get("associated_light_above_stopline"),
            "associated_light_color": payload.get("associated_light_color"),
            "visual_red_approach_target_mps": payload.get("visual_red_approach_target_mps"),
            "visual_dist_filtered": payload.get("visual_dist_filtered"),
            "tl_state_machine_state": payload.get("tl_state_machine_state"),
            "startup_phase": payload.get("startup_phase"),
            "throttle_startup_phase": payload.get("throttle_startup_phase"),
            "lane_transition_detected": payload.get("lane_transition_detected"),
            "route_target_jump_distance": payload.get("route_target_jump_distance"),
            "is_junction": payload.get("is_junction"),
            "upcoming_turn_type": payload.get("upcoming_turn_type"),
            "upcoming_turn_distance_m": payload.get("upcoming_turn_distance_m"),
            "turn_slowdown_active": payload.get("turn_slowdown_active"),
            "turn_speed_limit_mps": payload.get("turn_speed_limit_mps"),
            "turn_state": payload.get("turn_state"),
            "speed_state": payload.get("speed_state"),
            "speed_setpoint_raw": payload.get("speed_setpoint_raw"),
            "speed_setpoint_smoothed": payload.get("speed_setpoint_smoothed"),
            "cruise_allowed": payload.get("cruise_allowed"),
            "cte_recovery_active": payload.get("cte_recovery_active"),
            "junction_offroute_safety_stop": payload.get("junction_offroute_safety_stop"),
            "off_route": payload.get("off_route"),
            "safety_reason": payload.get("safety_reason"),
            "steer_raw": payload.get("steer_raw"),
            "steer_smoothed": payload.get("steer_smoothed"),
            "steer_rate_limited_value": payload.get("steer_rate_limited_value"),
            "steer_rate_limited": payload.get("steer_rate_limited"),
            "steer_cap_reason": payload.get("steer_cap_reason"),
            "lane_plan_age_ms": payload.get("lane_plan_age_ms"),
            "mission_status_age_ms": payload.get("mission_status_age_ms"),
            "route_age_ms": payload.get("route_age_ms"),
            "ego_status_age_ms": payload.get("ego_status_age_ms"),
            "control_cmd_age_ms": payload.get("control_cmd_age_ms"),
            "timeout_source": payload.get("timeout_source"),
            "timeout_threshold_ms": payload.get("timeout_threshold_ms"),
            "throttle": payload.get("throttle"),
            "brake": payload.get("brake"),
            "steer": payload.get("steer"),
            "reverse": payload.get("reverse"),
            "hand_brake": payload.get("hand_brake"),
            "vehicle_control": {
                "throttle": payload.get("throttle"),
                "brake": payload.get("brake"),
                "steer": payload.get("steer"),
                "reverse": payload.get("reverse"),
                "hand_brake": payload.get("hand_brake"),
            },
            "applied": payload.get("applied"),
            "reason": payload.get("reason"),
        }
        self.runtime_logger.write(record)

        now = time.time()
        if now - self.last_ros_log_s >= self.ros_log_period_s:
            self.last_ros_log_s = now
            self.get_logger().info(
                "adapter "
                f"applied={record['applied']} reason={record['reason']} "
                f"thr={record['throttle']} brake={record['brake']} "
                f"red_decel_brake_override={record['red_decel_brake_override']} "
                f"red_decel_brake_override_reason={record['red_decel_brake_override_reason']} "
                f"downhill_red_brake_guard={record['downhill_red_brake_guard']} "
                f"brake_released_for_visual_approach={record['brake_released_for_visual_approach']} "
                f"red_visual_slow_active_brake={record['red_visual_slow_active_brake']} "
                f"red_visual_slow_active_brake_reason={record['red_visual_slow_active_brake_reason']} "
                f"red_visual_slow_brake_value={record['red_visual_slow_brake_value']} "
                f"red_stop_hard_brake={record['red_stop_hard_brake']} "
                f"red_stop_hard_brake_reason={record['red_stop_hard_brake_reason']} "
                f"red_stop_hard_brake_value={record['red_stop_hard_brake_value']} "
                f"red_stop_2m_full_brake={record['red_stop_2m_full_brake']} "
                f"red_stop_2m_full_brake_value={record['red_stop_2m_full_brake_value']} "
                f"green_release_brake_cleared={record['green_release_brake_cleared']} "
                f"brake_release_reason={record['brake_release_reason']} "
                f"steer={record['steer']} reverse={record['reverse']} "
                f"junction_offroute_safety_stop={record['junction_offroute_safety_stop']} "
                f"off_route={record['off_route']} safety_reason={record['safety_reason']} "
                f"cmd_age={record['control_cmd_age_ms']}ms timeout={record['timeout_source']}"
            )


def main(args=None):
    rclpy.init(args=args)
    node = CarlaControlAdapterNode()

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
