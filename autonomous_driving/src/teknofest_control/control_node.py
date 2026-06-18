#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import time
from typing import Optional

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from teknofest_common.runtime_logging import RuntimeJsonlLogger
from teknofest_control.pure_pursuit import (
    PurePursuit,
    PurePursuitConfig,
    TargetPoint,
    VehiclePose,
)
from teknofest_control.speed_pid import SpeedPid, SpeedPidConfig


class ControlNode(Node):
    def __init__(self):
        super().__init__("control_node")

        # -------------------------
        # Config / parameter block
        # -------------------------
        self.declare_parameter("lane_plan_topic", "/adas/planning/lane_plan")
        self.declare_parameter("status_topic", "/adas/carla/status")
        self.declare_parameter("vehicle_command_topic", "/adas/control/vehicle_command")

        self.declare_parameter("control_period_s", 0.05)
        self.declare_parameter("plan_timeout_s", 1.8)
        self.declare_parameter("status_timeout_s", 3.0)
        self.declare_parameter("full_stop_speed_mps", 0.15)
        self.declare_parameter("hold_brake", 0.65)
        self.declare_parameter("max_throttle", 0.55)
        self.declare_parameter("max_brake", 0.85)
        self.declare_parameter("launch_throttle", 0.22)
        self.declare_parameter("brake_release_target_speed_mps", 0.5)

        self.declare_parameter("pid_kp", 0.32)
        self.declare_parameter("pid_ki", 0.04)
        self.declare_parameter("pid_kd", 0.03)
        self.declare_parameter("pid_integral_limit", 8.0)
        self.declare_parameter("accel_limit_per_s", 0.45)
        self.declare_parameter("decel_limit_per_s", 0.70)

        self.declare_parameter("wheel_base_m", 2.85)
        self.declare_parameter("max_steer_rad", 0.70)
        self.declare_parameter("steer_sign", 1.0)
        self.declare_parameter("max_steer_delta_per_s", 1.4)
        self.declare_parameter("steer_low_pass_alpha", 0.35)
        self.declare_parameter("startup_steer_limit_s", 5.0)
        self.declare_parameter("startup_max_steer", 0.14)
        self.declare_parameter("startup_lane_jump_max_steer", 0.08)
        self.declare_parameter("high_heading_max_steer", 0.25)
        self.declare_parameter("high_heading_limit_deg", 25.0)
        self.declare_parameter("turn_max_steer", 0.28)
        self.declare_parameter("junction_turn_max_steer", 0.36)
        self.declare_parameter("startup_throttle_limit_s", 3.0)
        self.declare_parameter("startup_max_throttle", 0.28)
        self.declare_parameter("low_speed_turn_max_throttle", 0.40)
        self.declare_parameter("coast_overspeed_margin_mps", 0.35)
        self.declare_parameter("speed_setpoint_accel_mps2", 1.1)
        self.declare_parameter("speed_setpoint_decel_mps2", 1.0)
        self.declare_parameter("throttle_slew_rate_per_s", 1.0)
        self.declare_parameter("log_root", "autonomous_driving/outputs/teknofest_sim_logs")
        self.declare_parameter("log_session_id", "")
        self.declare_parameter("jsonl_logging_enabled", True)
        self.declare_parameter("ros_log_period_s", 1.0)

        self.control_period_s = float(self.get_parameter("control_period_s").value)
        self.plan_timeout_s = float(self.get_parameter("plan_timeout_s").value)
        self.status_timeout_s = float(self.get_parameter("status_timeout_s").value)
        self.full_stop_speed_mps = float(self.get_parameter("full_stop_speed_mps").value)
        self.hold_brake = float(self.get_parameter("hold_brake").value)
        self.max_throttle = float(self.get_parameter("max_throttle").value)
        self.max_brake = float(self.get_parameter("max_brake").value)
        self.launch_throttle = float(self.get_parameter("launch_throttle").value)
        self.brake_release_target_speed_mps = float(
            self.get_parameter("brake_release_target_speed_mps").value
        )
        self.ros_log_period_s = float(self.get_parameter("ros_log_period_s").value)

        self.speed_pid = SpeedPid(
            SpeedPidConfig(
                kp=float(self.get_parameter("pid_kp").value),
                ki=float(self.get_parameter("pid_ki").value),
                kd=float(self.get_parameter("pid_kd").value),
                integral_limit=float(self.get_parameter("pid_integral_limit").value),
                accel_limit_per_s=float(self.get_parameter("accel_limit_per_s").value),
                decel_limit_per_s=float(self.get_parameter("decel_limit_per_s").value),
            )
        )
        self.pure_pursuit = PurePursuit(
            PurePursuitConfig(
                wheel_base_m=float(self.get_parameter("wheel_base_m").value),
                max_steer_rad=float(self.get_parameter("max_steer_rad").value),
                steer_sign=float(self.get_parameter("steer_sign").value),
                max_steer_delta_per_s=float(
                    self.get_parameter("max_steer_delta_per_s").value
                ),
                steer_low_pass_alpha=float(
                    self.get_parameter("steer_low_pass_alpha").value
                ),
            )
        )
        self.startup_steer_limit_s = float(self.get_parameter("startup_steer_limit_s").value)
        self.startup_max_steer = float(self.get_parameter("startup_max_steer").value)
        self.startup_lane_jump_max_steer = float(
            self.get_parameter("startup_lane_jump_max_steer").value
        )
        self.high_heading_max_steer = float(self.get_parameter("high_heading_max_steer").value)
        self.high_heading_limit_deg = float(self.get_parameter("high_heading_limit_deg").value)
        self.turn_max_steer = float(self.get_parameter("turn_max_steer").value)
        self.junction_turn_max_steer = float(self.get_parameter("junction_turn_max_steer").value)
        self.startup_throttle_limit_s = float(self.get_parameter("startup_throttle_limit_s").value)
        self.startup_max_throttle = float(self.get_parameter("startup_max_throttle").value)
        self.low_speed_turn_max_throttle = float(
            self.get_parameter("low_speed_turn_max_throttle").value
        )
        self.coast_overspeed_margin_mps = float(
            self.get_parameter("coast_overspeed_margin_mps").value
        )
        self.speed_setpoint_accel_mps2 = float(
            self.get_parameter("speed_setpoint_accel_mps2").value
        )
        self.speed_setpoint_decel_mps2 = float(
            self.get_parameter("speed_setpoint_decel_mps2").value
        )
        self.throttle_slew_rate_per_s = float(self.get_parameter("throttle_slew_rate_per_s").value)

        # -------------------------
        # Runtime state block
        # -------------------------
        self.plan_payload: Optional[dict] = None
        self.status_payload: Optional[dict] = None
        self.last_plan_time_s = 0.0
        self.last_status_time_s = 0.0
        self.last_tick_s = time.time()
        self.last_stop_request = False
        self.last_ros_log_s = 0.0
        self.last_command_payload: Optional[dict] = None
        self.started_at_s = time.time()
        self.speed_setpoint_smoothed = 0.0
        self.previous_throttle = 0.0

        self.runtime_logger = RuntimeJsonlLogger(
            node_name="control_node",
            file_name="control.jsonl",
            log_root=str(self.get_parameter("log_root").value),
            session_id=str(self.get_parameter("log_session_id").value) or None,
            enabled=bool(self.get_parameter("jsonl_logging_enabled").value),
        )

        # -------------------------
        # Publisher block
        # -------------------------
        self.command_pub = self.create_publisher(
            String,
            str(self.get_parameter("vehicle_command_topic").value),
            10,
        )

        # -------------------------
        # Subscriber block
        # -------------------------
        self.create_subscription(
            String,
            str(self.get_parameter("lane_plan_topic").value),
            self.plan_cb,
            10,
        )
        self.create_subscription(
            String,
            str(self.get_parameter("status_topic").value),
            self.status_cb,
            10,
        )

        # -------------------------
        # Timer block
        # -------------------------
        self.create_timer(self.control_period_s, self.tick)
        self.get_logger().info("Control node ready.")

    # -------------------------
    # Subscriber callbacks
    # -------------------------
    def plan_cb(self, msg: String):
        try:
            self.plan_payload = json.loads(msg.data)
            self.last_plan_time_s = time.time()
        except json.JSONDecodeError as exc:
            self.get_logger().warning(f"Invalid lane plan JSON ignored: {exc}")

    def status_cb(self, msg: String):
        try:
            self.status_payload = json.loads(msg.data)
            self.last_status_time_s = time.time()
        except json.JSONDecodeError as exc:
            self.get_logger().warning(f"Invalid CARLA status JSON ignored: {exc}")

    # -------------------------
    # Data conversion functions
    # -------------------------
    def current_speed_mps(self) -> float:
        if not self.status_payload:
            return 0.0

        try:
            return float(self.status_payload.get("speed_mps", 0.0))
        except (TypeError, ValueError):
            return 0.0

    def current_pose(self) -> Optional[VehiclePose]:
        status = self.status_payload or {}
        location = status.get("location") or {}
        rotation = status.get("rotation") or {}

        try:
            return VehiclePose(
                x=float(location["x"]),
                y=float(location["y"]),
                yaw_deg=float(rotation.get("yaw", 0.0)),
            )
        except (KeyError, TypeError, ValueError):
            return None

    def target_point(self) -> Optional[TargetPoint]:
        point = (self.plan_payload or {}).get("target_point") or {}

        try:
            return TargetPoint(x=float(point["x"]), y=float(point["y"]))
        except (KeyError, TypeError, ValueError):
            return None

    # -------------------------
    # Control decision functions
    # -------------------------
    def clamp(self, value: float, low: float, high: float) -> float:
        return max(low, min(high, value))

    def stale(self, now: float) -> Optional[str]:
        if self.plan_payload is None or now - self.last_plan_time_s > self.plan_timeout_s:
            return "plan_timeout"

        if self.status_payload is None or now - self.last_status_time_s > self.status_timeout_s:
            return "status_timeout"

        return None

    def tick(self):
        now = time.time()
        dt = max(1e-3, now - self.last_tick_s)
        self.last_tick_s = now

        stale_reason = self.stale(now)
        if stale_reason:
            self.speed_pid.reset()
            self.publish_command(
                throttle=0.0,
                brake=self.hold_brake,
                steer=0.0,
                target_speed_mps=0.0,
                reason=stale_reason,
                timeout_source=stale_reason,
            )
            return

        pose = self.current_pose()
        target = self.target_point()
        current_speed = self.current_speed_mps()

        if pose is None or target is None:
            self.publish_command(
                throttle=0.0,
                brake=self.hold_brake,
                steer=0.0,
                target_speed_mps=0.0,
                reason="invalid_pose_or_target",
                timeout_source="invalid_pose_or_target",
            )
            return

        target_speed_raw = float((self.plan_payload or {}).get("target_speed_mps", 0.0))
        target_speed = self.smooth_speed_setpoint(target_speed_raw, dt)
        stop_request = bool((self.plan_payload or {}).get("stop_request", False))
        steer = self.pure_pursuit.compute(pose, target, dt)
        steer_cap_reason = ""
        steer_cap_limit = 1.0
        startup_phase = now - self.started_at_s <= self.startup_steer_limit_s
        throttle_startup_phase = now - self.started_at_s <= self.startup_throttle_limit_s
        heading_error = float((self.plan_payload or {}).get("heading_error_deg") or 0.0)
        turn_slowdown_active = bool((self.plan_payload or {}).get("turn_slowdown_active", False))
        turn_state = str((self.plan_payload or {}).get("turn_state") or "")
        plan_reason = str((self.plan_payload or {}).get("reason", "track_lane"))
        plan_stop_reason = str((self.plan_payload or {}).get("stop_reason") or "")
        tl_decision_reason = str((self.plan_payload or {}).get("tl_decision_reason") or "")
        tl_state_machine_state = str((self.plan_payload or {}).get("tl_state_machine_state") or "")
        associated_light_color = str((self.plan_payload or {}).get("associated_light_color") or "").lower()
        red_reason_text = " ".join((plan_reason, plan_stop_reason, tl_decision_reason)).lower()
        green_release_active = bool(
            plan_reason == "green_release"
            or tl_decision_reason == "green_release"
            or tl_state_machine_state == "GREEN_RELEASE"
            or associated_light_color == "green"
        )
        if green_release_active:
            stop_request = False
        visual_red_brake = bool((self.plan_payload or {}).get("visual_red_brake", False))
        visual_red_commit = bool((self.plan_payload or {}).get("visual_red_commit", False))
        visual_red_approach = bool((self.plan_payload or {}).get("visual_red_approach", False))
        associated_light_above_stopline = bool(
            (self.plan_payload or {}).get("associated_light_above_stopline", False)
        )
        visual_dist_filtered = (self.plan_payload or {}).get("visual_dist_filtered")
        try:
            visual_dist_filtered_f = (
                float(visual_dist_filtered)
                if visual_dist_filtered is not None
                else None
            )
        except (TypeError, ValueError):
            visual_dist_filtered_f = None
        red_decel_brake_override = False
        downhill_red_brake_guard = False
        red_decel_brake_override_reason = ""
        brake_released_for_visual_approach = False
        red_visual_slow_active_brake = False
        red_visual_slow_active_brake_reason = ""
        red_visual_slow_brake_value = 0.0
        red_stop_hard_brake = False
        red_stop_hard_brake_reason = ""
        red_stop_hard_brake_value = 0.0
        red_stop_2m_full_brake = False
        red_stop_2m_full_brake_value = 0.0
        green_release_brake_cleared = False
        brake_release_reason = ""
        junction_offroute_safety_stop = bool(
            (self.plan_payload or {}).get("junction_offroute_safety_stop", False)
        )
        if junction_offroute_safety_stop:
            target_speed_raw = 0.0
            target_speed = 0.0
            self.speed_setpoint_smoothed = 0.0
        junction_turn_active = turn_state in {"ENTERING_JUNCTION", "IN_JUNCTION_TURN", "EXITING_JUNCTION"}
        lane_jump_rejected = bool((self.plan_payload or {}).get("lane_jump_rejected", False))

        if junction_offroute_safety_stop:
            previous_steer = float((self.last_command_payload or {}).get("steer", 0.0))
            if abs(steer) > abs(previous_steer):
                steer = math.copysign(abs(previous_steer), steer) if previous_steer != 0.0 else 0.0
                self.pure_pursuit.force_previous_steer(steer)
                steer_cap_reason = "junction_offroute_safety_steer_hold"
                steer_cap_limit = abs(previous_steer)

        if startup_phase and lane_jump_rejected and abs(steer) > self.startup_lane_jump_max_steer:
            steer = self.clamp(steer, -self.startup_lane_jump_max_steer, self.startup_lane_jump_max_steer)
            steer_cap_reason = "startup_lane_jump_steer_cap"
            steer_cap_limit = self.startup_lane_jump_max_steer
        elif startup_phase and abs(steer) > self.startup_max_steer:
            steer = self.clamp(steer, -self.startup_max_steer, self.startup_max_steer)
            steer_cap_reason = "startup_steer_cap"
            steer_cap_limit = self.startup_max_steer
        elif junction_turn_active and abs(steer) > self.junction_turn_max_steer:
            steer = self.clamp(steer, -self.junction_turn_max_steer, self.junction_turn_max_steer)
            steer_cap_reason = "junction_turn_steer_cap"
            steer_cap_limit = self.junction_turn_max_steer
        elif turn_slowdown_active and abs(steer) > self.turn_max_steer:
            steer = self.clamp(steer, -self.turn_max_steer, self.turn_max_steer)
            steer_cap_reason = "turn_steer_cap"
            steer_cap_limit = self.turn_max_steer
        elif abs(heading_error) >= self.high_heading_limit_deg and abs(steer) > self.high_heading_max_steer:
            steer = self.clamp(steer, -self.high_heading_max_steer, self.high_heading_max_steer)
            steer_cap_reason = "heading_steer_cap"
            steer_cap_limit = self.high_heading_max_steer

        if steer_cap_reason:
            self.pure_pursuit.force_previous_steer(steer)

        if self.last_stop_request and not stop_request:
            self.speed_pid.reset()
        self.last_stop_request = stop_request

        if stop_request and current_speed <= self.full_stop_speed_mps:
            self.speed_pid.reset()
            throttle = 0.0
            brake = self.hold_brake
            self.previous_throttle = 0.0
            if "red" in red_reason_text:
                reason = plan_stop_reason or tl_decision_reason or "hold_stop"
                downhill_red_brake_guard = True
                red_stop_full_brake_request = bool(
                    "red_stop_2m_visual_stopline" in red_reason_text
                    or "red_stop_2m_full_brake" in red_reason_text
                    or "red_visual_near_light_failsafe_stop" in red_reason_text
                    or "locked_red_stop_point_overrun" in red_reason_text
                    or tl_state_machine_state == "RED_STOP_COMMIT"
                    or "red_stop_commit" in red_reason_text
                )
                if red_stop_full_brake_request:
                    brake = 1.0
                    red_stop_2m_full_brake = True
                    red_stop_2m_full_brake_value = 1.0
                    red_stop_hard_brake = True
                    red_stop_hard_brake_reason = "red_stop_2m_full_brake"
                    red_stop_hard_brake_value = 1.0
            else:
                reason = "hold_stop"
        else:
            accel_command = self.speed_pid.step(target_speed, current_speed, dt)
            throttle, brake = self.speed_pid.split_throttle_brake(accel_command)
            throttle = self.clamp(throttle, 0.0, self.max_throttle)
            brake = self.clamp(brake, 0.0, self.max_brake)
            reason = plan_reason

            if stop_request:
                throttle = 0.0
                red_stop_request = "red" in red_reason_text
                red_stop_full_brake_request = bool(
                    red_stop_request
                    and (
                        "red_stop_2m_visual_stopline" in red_reason_text
                        or "red_stop_2m_full_brake" in red_reason_text
                        or "red_visual_near_light_failsafe_stop" in red_reason_text
                        or "locked_red_stop_point_overrun" in red_reason_text
                        or tl_state_machine_state == "RED_STOP_COMMIT"
                        or "red_stop_commit" in red_reason_text
                    )
                )
                if red_stop_full_brake_request:
                    requested_stop_brake = 1.0
                    red_stop_2m_full_brake = True
                    red_stop_2m_full_brake_value = 1.0
                    red_stop_hard_brake = True
                    red_stop_hard_brake_reason = "red_stop_2m_full_brake"
                    red_stop_hard_brake_value = 1.0
                elif red_stop_request and current_speed > 0.2:
                    requested_stop_brake = 1.0
                    red_stop_hard_brake = True
                    red_stop_hard_brake_reason = "moving_red_stop_request"
                    red_stop_hard_brake_value = requested_stop_brake
                else:
                    requested_stop_brake = (
                        self.hold_brake
                        if red_stop_request
                        else 0.45
                        if junction_offroute_safety_stop
                        else 0.35
                    )
                brake = max(brake, self.clamp(float(requested_stop_brake), 0.0, 1.0))
                reason = plan_stop_reason or "requested_stop"
                if red_stop_request:
                    downhill_red_brake_guard = True
                if junction_offroute_safety_stop:
                    reason = "junction_offroute_safety_stop"
            elif target_speed > self.brake_release_target_speed_mps:
                if current_speed < target_speed + 0.25:
                    brake = 0.0
                if current_speed < 0.45 and throttle < self.launch_throttle:
                    throttle = min(self.max_throttle, self.launch_throttle)

            if not stop_request and current_speed > target_speed + self.coast_overspeed_margin_mps:
                brake = 0.0
                throttle = 0.0

            if (
                not stop_request
                and plan_reason in {"turn_approach", "turn_entry"}
                and current_speed > target_speed
            ):
                brake = 0.0
                throttle = 0.0
                reason = f"{plan_reason}_coast"

            if throttle_startup_phase and not stop_request:
                throttle = min(throttle, self.startup_max_throttle)

            if turn_slowdown_active and abs(steer) > 0.18 and not stop_request:
                throttle = min(throttle, self.low_speed_turn_max_throttle)

            red_visual_slow_reason_active = (
                plan_reason == "red_visual_slow_10m_rule"
                or tl_decision_reason == "red_visual_slow_10m_rule"
            )
            red_decel_mode = bool(
                plan_reason in {"red_decel", "visual_red_approach_to_stopline", "red_visual_slow_10m_rule"}
                or tl_decision_reason in {"red_decel", "visual_red_approach_to_stopline", "red_visual_slow_10m_rule"}
                or visual_red_approach
                or visual_red_brake
            )
            visual_commit_window = bool(
                visual_dist_filtered_f is not None
                and visual_dist_filtered_f <= 3.0
            )
            visual_near_light_window = bool(
                visual_dist_filtered_f is not None
                and visual_dist_filtered_f <= 6.5
            )
            visual_approach_cruise = bool(
                visual_red_approach
                and not visual_red_commit
                and not visual_commit_window
                and not stop_request
            )

            if red_decel_mode and current_speed > target_speed + 0.25:
                throttle = 0.0
                strong_red_slow_brake = bool(
                    downhill_red_brake_guard
                    or associated_light_above_stopline
                    or visual_near_light_window
                )
                requested_brake = 0.60 if strong_red_slow_brake else 0.45
                brake = max(brake, min(self.max_brake, requested_brake))
                red_decel_brake_override = True
                downhill_red_brake_guard = bool(downhill_red_brake_guard or visual_red_brake)
                red_visual_slow_active_brake = True
                red_visual_slow_active_brake_reason = (
                    "red_visual_slow_active_brake"
                    if red_visual_slow_reason_active
                    else "visual_red_active_brake"
                )
                red_visual_slow_brake_value = max(red_visual_slow_brake_value, min(self.max_brake, requested_brake))
                red_decel_brake_override_reason = (
                    "red_visual_slow_active_brake"
                    if red_visual_slow_reason_active
                    else "visual_commit_window_overspeed"
                    if visual_commit_window
                    else "red_visual_near_light_active_brake"
                    if strong_red_slow_brake
                    else "red_decel_overspeed"
                )

            if red_decel_mode and visual_red_brake and visual_commit_window:
                throttle = 0.0
                brake = max(brake, min(self.max_brake, 0.35))
                red_decel_brake_override = True
                downhill_red_brake_guard = True
                red_decel_brake_override_reason = red_decel_brake_override_reason or "visual_commit_window"

            if visual_approach_cruise and current_speed <= target_speed + 0.25:
                brake = 0.0
                throttle = min(throttle, 0.12)
                brake_released_for_visual_approach = True
                if throttle > 0.0:
                    reason = "visual_red_approach_to_stopline"

            if stop_request and "red" in red_reason_text:
                throttle = 0.0
                red_stop_full_brake_request = bool(
                    "red_stop_2m_visual_stopline" in red_reason_text
                    or "red_stop_2m_full_brake" in red_reason_text
                    or "red_visual_near_light_failsafe_stop" in red_reason_text
                    or "locked_red_stop_point_overrun" in red_reason_text
                    or tl_state_machine_state == "RED_STOP_COMMIT"
                    or "red_stop_commit" in red_reason_text
                )
                requested_stop_brake = 1.0 if red_stop_full_brake_request else 1.0 if current_speed > 0.2 else self.hold_brake
                brake = max(brake, self.clamp(float(requested_stop_brake), 0.0, 1.0))
                downhill_red_brake_guard = True
                if red_stop_full_brake_request:
                    red_stop_2m_full_brake = True
                    red_stop_2m_full_brake_value = 1.0
                if current_speed > 0.2 or red_stop_full_brake_request:
                    red_stop_hard_brake = True
                    red_stop_hard_brake_reason = (
                        "red_stop_2m_full_brake"
                        if red_stop_full_brake_request
                        else
                        "red_visual_near_light_failsafe_stop"
                        if "red_visual_near_light_failsafe_stop" in red_reason_text
                        else "red_stop_2m_visual_stopline"
                        if "red_stop_2m_visual_stopline" in red_reason_text
                        else "locked_red_stop_point_overrun"
                        if "locked_red_stop_point_overrun" in red_reason_text
                        else "moving_red_stop_request"
                    )
                    red_stop_hard_brake_value = max(red_stop_hard_brake_value, self.clamp(float(requested_stop_brake), 0.0, 1.0))
                red_decel_brake_override_reason = red_decel_brake_override_reason or "red_stop_request"

            throttle = self.slew_throttle(throttle, dt)

            if throttle > 0.0 and brake > 0.0:
                if stop_request:
                    throttle = 0.0
                else:
                    brake = 0.0

        if green_release_active:
            brake = 0.0
            red_decel_brake_override = False
            downhill_red_brake_guard = False
            red_decel_brake_override_reason = ""
            brake_released_for_visual_approach = False
            red_visual_slow_active_brake = False
            red_visual_slow_active_brake_reason = ""
            red_visual_slow_brake_value = 0.0
            red_stop_hard_brake = False
            red_stop_hard_brake_reason = ""
            red_stop_hard_brake_value = 0.0
            red_stop_2m_full_brake = False
            red_stop_2m_full_brake_value = 0.0
            green_release_brake_cleared = True
            brake_release_reason = "green_release"

        self.publish_command(
            throttle=throttle,
            brake=brake,
            steer=steer,
            target_speed_mps=target_speed,
            target_speed_raw_mps=target_speed_raw,
            reason=reason,
            timeout_source="",
            steer_cap_reason=steer_cap_reason,
            steer_cap_limit=steer_cap_limit,
            startup_phase=startup_phase,
            throttle_startup_phase=throttle_startup_phase,
            red_decel_brake_override=red_decel_brake_override,
            downhill_red_brake_guard=downhill_red_brake_guard,
            red_decel_brake_override_reason=red_decel_brake_override_reason,
            brake_released_for_visual_approach=brake_released_for_visual_approach,
            red_visual_slow_active_brake=red_visual_slow_active_brake,
            red_visual_slow_active_brake_reason=red_visual_slow_active_brake_reason,
            red_visual_slow_brake_value=red_visual_slow_brake_value,
            red_stop_hard_brake=red_stop_hard_brake,
            red_stop_hard_brake_reason=red_stop_hard_brake_reason,
            red_stop_hard_brake_value=red_stop_hard_brake_value,
            red_stop_2m_full_brake=red_stop_2m_full_brake,
            red_stop_2m_full_brake_value=red_stop_2m_full_brake_value,
            green_release_brake_cleared=green_release_brake_cleared,
            brake_release_reason=brake_release_reason,
            command_stop_request=stop_request,
        )

    # -------------------------
    # Debug / publish block
    # -------------------------
    def publish_command(
        self,
        *,
        throttle: float,
        brake: float,
        steer: float,
        target_speed_mps: float,
        target_speed_raw_mps: float = 0.0,
        reason: str,
        timeout_source: str,
        steer_cap_reason: str = "",
        steer_cap_limit: float = 1.0,
        startup_phase: bool = False,
        throttle_startup_phase: bool = False,
        red_decel_brake_override: bool = False,
        downhill_red_brake_guard: bool = False,
        red_decel_brake_override_reason: str = "",
        brake_released_for_visual_approach: bool = False,
        red_visual_slow_active_brake: bool = False,
        red_visual_slow_active_brake_reason: str = "",
        red_visual_slow_brake_value: float = 0.0,
        red_stop_hard_brake: bool = False,
        red_stop_hard_brake_reason: str = "",
        red_stop_hard_brake_value: float = 0.0,
        red_stop_2m_full_brake: bool = False,
        red_stop_2m_full_brake_value: float = 0.0,
        green_release_brake_cleared: bool = False,
        brake_release_reason: str = "",
        command_stop_request: bool | None = None,
    ):
        pose = self.current_pose()
        now = time.time()
        lane_plan_age_ms = self.age_ms(now, self.last_plan_time_s)
        ego_status_age_ms = self.age_ms(now, self.last_status_time_s)
        mission_status_age_ms = (self.plan_payload or {}).get("mission_status_age_ms")
        route_age_ms = (self.plan_payload or {}).get("route_age_ms")
        payload = {
            "stamp": time.time(),
            "source": "control_node",
            "ego_x": round(float(pose.x), 4) if pose else None,
            "ego_y": round(float(pose.y), 4) if pose else None,
            "ego_yaw": round(float(pose.yaw_deg), 4) if pose else None,
            "throttle": round(self.clamp(float(throttle), 0.0, 1.0), 4),
            "brake": round(self.clamp(float(brake), 0.0, 1.0), 4),
            "steer": round(self.clamp(float(steer), -1.0, 1.0), 4),
            "steer_raw": round(float(self.pure_pursuit.last_steer_raw), 4),
            "steer_smoothed": round(float(self.pure_pursuit.last_steer_smoothed), 4),
            "steer_rate_limited_value": round(float(self.pure_pursuit.last_steer_rate_limited), 4),
            "steer_rate_limited": bool(self.pure_pursuit.last_rate_limited),
            "steer_cap_reason": steer_cap_reason,
            "steer_cap_limit": round(float(steer_cap_limit), 3),
            "reverse": False,
            "hand_brake": False,
            "target_speed_mps": round(float(target_speed_mps), 3),
            "speed_setpoint_raw": round(float(target_speed_raw_mps), 3),
            "speed_setpoint_smoothed": round(float(target_speed_mps), 3),
            "current_speed_mps": round(self.current_speed_mps(), 3),
            "lane_plan_age_ms": lane_plan_age_ms,
            "mission_status_age_ms": mission_status_age_ms,
            "route_age_ms": route_age_ms,
            "ego_status_age_ms": ego_status_age_ms,
            "timeout_source": timeout_source,
            "startup_phase": bool(startup_phase),
            "throttle_startup_phase": bool(throttle_startup_phase),
            "timeout_threshold_ms": {
                "lane_plan": int(self.plan_timeout_s * 1000.0),
                "ego_status": int(self.status_timeout_s * 1000.0),
            },
            "stop_request": bool(command_stop_request)
            if command_stop_request is not None
            else bool((self.plan_payload or {}).get("stop_request", False)),
            "active_mission_target": ((self.plan_payload or {}).get("active_mission_target")),
            "mission_state": ((self.plan_payload or {}).get("mission_state")),
            "route_index": ((self.plan_payload or {}).get("route_index")),
            "route_length": ((self.plan_payload or {}).get("route_length")),
            "lane_target_x": (((self.plan_payload or {}).get("target_point") or {}).get("x")),
            "lane_target_y": (((self.plan_payload or {}).get("target_point") or {}).get("y")),
            "cross_track_error": ((self.plan_payload or {}).get("lateral_error_m")),
            "heading_error": ((self.plan_payload or {}).get("heading_error_deg")),
            "lane_transition_detected": ((self.plan_payload or {}).get("lane_transition_detected")),
            "route_target_jump_distance": ((self.plan_payload or {}).get("route_target_jump_distance")),
            "is_junction": ((self.plan_payload or {}).get("is_junction")),
            "upcoming_turn_type": ((self.plan_payload or {}).get("upcoming_turn_type")),
            "upcoming_turn_distance_m": ((self.plan_payload or {}).get("upcoming_turn_distance_m")),
            "turn_slowdown_active": ((self.plan_payload or {}).get("turn_slowdown_active")),
            "turn_speed_limit_mps": ((self.plan_payload or {}).get("turn_speed_limit_mps")),
            "turn_state": ((self.plan_payload or {}).get("turn_state")),
            "speed_state": ((self.plan_payload or {}).get("speed_state")),
            "cruise_allowed": ((self.plan_payload or {}).get("cruise_allowed")),
            "cte_recovery_active": ((self.plan_payload or {}).get("cte_recovery_active")),
            "junction_offroute_safety_stop": ((self.plan_payload or {}).get("junction_offroute_safety_stop")),
            "off_route": ((self.plan_payload or {}).get("off_route")),
            "safety_reason": ((self.plan_payload or {}).get("safety_reason")),
            "reason": reason,
            "red_decel_brake_override": bool(red_decel_brake_override),
            "downhill_red_brake_guard": bool(downhill_red_brake_guard),
            "red_decel_brake_override_reason": red_decel_brake_override_reason,
            "brake_released_for_visual_approach": bool(brake_released_for_visual_approach),
            "red_visual_slow_active_brake": bool(red_visual_slow_active_brake),
            "red_visual_slow_active_brake_reason": red_visual_slow_active_brake_reason,
            "red_visual_slow_brake_value": round(float(red_visual_slow_brake_value), 3),
            "red_stop_hard_brake": bool(red_stop_hard_brake),
            "red_stop_hard_brake_reason": red_stop_hard_brake_reason,
            "red_stop_hard_brake_value": round(float(red_stop_hard_brake_value), 3),
            "red_stop_2m_full_brake": bool(red_stop_2m_full_brake),
            "red_stop_2m_full_brake_value": round(float(red_stop_2m_full_brake_value), 3),
            "green_release_brake_cleared": bool(green_release_brake_cleared),
            "brake_release_reason": brake_release_reason,
            "visual_red_brake": bool((self.plan_payload or {}).get("visual_red_brake", False)),
            "visual_red_commit": bool((self.plan_payload or {}).get("visual_red_commit", False)),
            "visual_red_approach": bool((self.plan_payload or {}).get("visual_red_approach", False)),
            "associated_light_above_stopline": bool(
                (self.plan_payload or {}).get("associated_light_above_stopline", False)
            ),
            "associated_light_color": ((self.plan_payload or {}).get("associated_light_color")),
            "visual_red_approach_target_mps": ((self.plan_payload or {}).get("visual_red_approach_target_mps")),
            "visual_dist_filtered": ((self.plan_payload or {}).get("visual_dist_filtered")),
            "tl_state_machine_state": ((self.plan_payload or {}).get("tl_state_machine_state")),
            "tl_decision_reason": ((self.plan_payload or {}).get("tl_decision_reason")),
            "stop_reason": ((self.plan_payload or {}).get("stop_reason")),
        }
        msg = String()
        msg.data = json.dumps(payload, ensure_ascii=False)
        self.command_pub.publish(msg)
        self.last_command_payload = payload
        self.log_runtime(payload)

    def age_ms(self, now: float, stamp_s: float):
        if stamp_s <= 0.0:
            return None
        return int(max(0.0, now - stamp_s) * 1000.0)

    def smooth_speed_setpoint(self, raw_speed: float, dt: float) -> float:
        if self.speed_setpoint_smoothed <= 0.0:
            self.speed_setpoint_smoothed = raw_speed

        delta = raw_speed - self.speed_setpoint_smoothed
        max_delta = (
            self.speed_setpoint_accel_mps2 * dt
            if delta >= 0.0
            else self.speed_setpoint_decel_mps2 * dt
        )

        if abs(delta) <= max_delta:
            self.speed_setpoint_smoothed = raw_speed
        else:
            self.speed_setpoint_smoothed += math.copysign(max_delta, delta)

        return self.speed_setpoint_smoothed

    def slew_throttle(self, throttle: float, dt: float) -> float:
        if throttle <= self.previous_throttle:
            self.previous_throttle = self.clamp(throttle, 0.0, self.max_throttle)
            return self.previous_throttle

        max_delta = self.throttle_slew_rate_per_s * max(1e-3, dt)
        delta = min(throttle - self.previous_throttle, max_delta)
        out = self.clamp(self.previous_throttle + delta, 0.0, self.max_throttle)
        self.previous_throttle = out
        return out

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
            "speed_setpoint_raw": payload.get("speed_setpoint_raw"),
            "speed_setpoint_smoothed": payload.get("speed_setpoint_smoothed"),
            "speed_reason": payload.get("reason"),
            "stop_request": payload.get("stop_request"),
            "lane_plan_age_ms": payload.get("lane_plan_age_ms"),
            "mission_status_age_ms": payload.get("mission_status_age_ms"),
            "route_age_ms": payload.get("route_age_ms"),
            "ego_status_age_ms": payload.get("ego_status_age_ms"),
            "timeout_source": payload.get("timeout_source"),
            "timeout_threshold_ms": payload.get("timeout_threshold_ms"),
            "throttle": payload.get("throttle"),
            "brake": payload.get("brake"),
            "steer": payload.get("steer"),
            "steer_raw": payload.get("steer_raw"),
            "steer_smoothed": payload.get("steer_smoothed"),
            "steer_rate_limited_value": payload.get("steer_rate_limited_value"),
            "steer_rate_limited": payload.get("steer_rate_limited"),
            "steer_cap_reason": payload.get("steer_cap_reason"),
            "steer_cap_limit": payload.get("steer_cap_limit"),
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
            "cruise_allowed": payload.get("cruise_allowed"),
            "cte_recovery_active": payload.get("cte_recovery_active"),
            "junction_offroute_safety_stop": payload.get("junction_offroute_safety_stop"),
            "off_route": payload.get("off_route"),
            "safety_reason": payload.get("safety_reason"),
            "reverse": payload.get("reverse"),
            "hand_brake": payload.get("hand_brake"),
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
            "tl_decision_reason": payload.get("tl_decision_reason"),
            "stop_reason": payload.get("stop_reason"),
        }
        self.runtime_logger.write(record)

        now = time.time()
        if now - self.last_ros_log_s >= self.ros_log_period_s:
            self.last_ros_log_s = now
            self.get_logger().info(
                "control "
                f"state={record['mission_state']} target={record['active_mission_target']} "
                f"v={record['current_speed_mps']} target_v={record['target_speed_mps']} "
                f"stop={record['stop_request']} thr={record['throttle']} "
                f"brake={record['brake']} steer={record['steer']} reason={record['speed_reason']} "
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
                f"junction_offroute_safety_stop={record['junction_offroute_safety_stop']} "
                f"off_route={record['off_route']} safety_reason={record['safety_reason']} "
                f"timeout={record['timeout_source']} ages(plan/status)="
                f"{record['lane_plan_age_ms']}/{record['ego_status_age_ms']}ms"
            )


def main(args=None):
    rclpy.init(args=args)
    node = ControlNode()

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
