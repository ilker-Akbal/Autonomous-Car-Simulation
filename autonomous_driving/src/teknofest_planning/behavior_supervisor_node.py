#!/usr/bin/env python3
from __future__ import annotations

import json
import time
from enum import Enum, auto
from typing import Optional

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from teknofest_common.runtime_logging import RuntimeJsonlLogger
from teknofest_planning.route_event_finder import RouteEvent, RouteEventFinder
from teknofest_planning.speed_profile import SpeedProfilePlanner


class BehaviorState(Enum):
    NORMAL_DRIVE = auto()
    TL_APPROACH = auto()
    TL_STOP = auto()
    TL_WAIT_GREEN = auto()
    TL_RELEASE = auto()
    MISSION_STOP = auto()
    PARKING = auto()
    EMERGENCY_STOP = auto()


class BehaviorSupervisorNode(Node):
    def __init__(self):
        super().__init__("behavior_supervisor_node")

        self.declare_parameter("lane_plan_raw_topic", "/adas/planning/lane_plan_raw")
        self.declare_parameter("lane_plan_topic", "/adas/planning/lane_plan")
        self.declare_parameter("route_topic", "/adas/planning/route")
        self.declare_parameter("status_topic", "/adas/carla/status")
        self.declare_parameter("mission_topic", "/adas/teknofest/mission")
        self.declare_parameter("tl_event_topic", "/adas/planning/tl_event")
        self.declare_parameter("publish_period_s", 0.1)
        self.declare_parameter("stop_buffer_m", 1.0)
        self.declare_parameter("max_accel_mps2", 1.1)
        self.declare_parameter("max_decel_mps2", 1.4)
        self.declare_parameter("slow_speed_mps", 0.8)
        self.declare_parameter("follow_time_gap_s", 1.4)
        self.declare_parameter("route_corridor_width_m", 4.5)
        self.declare_parameter("status_timeout_s", 2.5)
        self.declare_parameter("lane_plan_timeout_s", 1.5)
        self.declare_parameter("log_root", "autonomous_driving/outputs/teknofest_sim_logs")
        self.declare_parameter("log_session_id", "")
        self.declare_parameter("jsonl_logging_enabled", True)
        self.declare_parameter("ros_log_period_s", 1.0)

        self.lane_plan_raw_payload: Optional[dict] = None
        self.route_payload: Optional[dict] = None
        self.status_payload: Optional[dict] = None
        self.mission_payload: Optional[dict] = None
        self.tl_event_payload: Optional[dict] = None
        self.last_lane_plan_raw_s = 0.0
        self.last_status_s = 0.0
        self.last_ros_log_s = 0.0
        self.state = BehaviorState.EMERGENCY_STOP
        self.blocking_tl_active = False

        self.speed_planner = SpeedProfilePlanner(
            max_accel_mps2=float(self.get_parameter("max_accel_mps2").value),
            max_decel_mps2=float(self.get_parameter("max_decel_mps2").value),
            slow_speed_mps=float(self.get_parameter("slow_speed_mps").value),
            stop_buffer_m=float(self.get_parameter("stop_buffer_m").value),
            follow_time_gap_s=float(self.get_parameter("follow_time_gap_s").value),
        )
        self.event_finder = RouteEventFinder(
            route_corridor_width_m=float(self.get_parameter("route_corridor_width_m").value)
        )
        self.runtime_logger = RuntimeJsonlLogger(
            node_name="behavior_supervisor_node",
            file_name="behavior_supervisor.jsonl",
            log_root=str(self.get_parameter("log_root").value),
            session_id=str(self.get_parameter("log_session_id").value) or None,
            enabled=bool(self.get_parameter("jsonl_logging_enabled").value),
        )

        self.plan_pub = self.create_publisher(
            String,
            str(self.get_parameter("lane_plan_topic").value),
            10,
        )
        self.create_subscription(
            String,
            str(self.get_parameter("lane_plan_raw_topic").value),
            self.lane_plan_raw_cb,
            10,
        )
        self.create_subscription(
            String,
            str(self.get_parameter("route_topic").value),
            self.route_cb,
            10,
        )
        self.create_subscription(
            String,
            str(self.get_parameter("status_topic").value),
            self.status_cb,
            10,
        )
        self.create_subscription(
            String,
            str(self.get_parameter("mission_topic").value),
            self.mission_cb,
            10,
        )
        self.create_subscription(
            String,
            str(self.get_parameter("tl_event_topic").value),
            self.tl_event_cb,
            10,
        )
        self.create_timer(float(self.get_parameter("publish_period_s").value), self.tick)
        self.get_logger().info("Behavior supervisor node ready.")

    def _parse_json(self, msg: String) -> Optional[dict]:
        try:
            return json.loads(msg.data)
        except (json.JSONDecodeError, TypeError):
            return None

    def lane_plan_raw_cb(self, msg: String):
        payload = self._parse_json(msg)
        if payload is not None:
            self.lane_plan_raw_payload = payload
            self.last_lane_plan_raw_s = time.time()

    def route_cb(self, msg: String):
        payload = self._parse_json(msg)
        if payload is not None:
            self.route_payload = payload

    def status_cb(self, msg: String):
        payload = self._parse_json(msg)
        if payload is not None:
            self.status_payload = payload
            self.last_status_s = time.time()

    def mission_cb(self, msg: String):
        payload = self._parse_json(msg)
        if payload is not None:
            self.mission_payload = payload

    def tl_event_cb(self, msg: String):
        payload = self._parse_json(msg)
        if payload is not None:
            self.tl_event_payload = payload

    def _current_speed_mps(self) -> float:
        try:
            return float((self.status_payload or {}).get("speed_mps", 0.0) or 0.0)
        except (TypeError, ValueError):
            return 0.0

    def _fallback_target_point(self) -> dict:
        location = (self.status_payload or {}).get("location") or {}
        rotation = (self.status_payload or {}).get("rotation") or {}
        return {
            "x": float(location.get("x", 0.0) or 0.0),
            "y": float(location.get("y", 0.0) or 0.0),
            "z": float(location.get("z", 0.0) or 0.0),
            "yaw_deg": float(rotation.get("yaw", 0.0) or 0.0),
        }

    def _stop_output(self, reason: str) -> dict:
        self.state = BehaviorState.EMERGENCY_STOP
        payload = dict(self.lane_plan_raw_payload or {})
        payload.setdefault("target_point", self._fallback_target_point())
        payload.update({
            "stamp": time.time(),
            "source": "behavior_supervisor_node",
            "behavior_state": self.state.name,
            "event_type": "EMERGENCY",
            "event_distance_m": None,
            "event_source": "behavior_supervisor_node",
            "event_color": "unknown",
            "speed_profile_reason": reason,
            "supervisor_reason": reason,
            "raw_target_speed_mps": float((self.lane_plan_raw_payload or {}).get("target_speed_mps", 0.0) or 0.0),
            "target_speed_mps": 0.0,
            "stop_request": True,
            "reason": reason,
            "stop_reason": reason,
        })
        return payload

    def _yellow_requires_stop(self, event: RouteEvent, current_speed_mps: float) -> bool:
        if event.distance_m is None:
            return True
        time_to_line = float(event.distance_m) / max(0.1, current_speed_mps)
        return not (time_to_line < 1.0 and current_speed_mps > 1.0)

    def _apply_behavior(self, event: RouteEvent, current_speed_mps: float, desired_speed_mps: float) -> tuple[BehaviorState, float, bool, str, str]:
        event_type = str(event.event_type or "NONE").upper()
        stop_required = False
        supervisor_reason = event.reason
        effective_event_type = event_type

        if event_type == "MISSION_STOP":
            self.blocking_tl_active = False
            return BehaviorState.MISSION_STOP, 0.0, True, "mission_stop", "mission_must_stop"

        if event_type == "PARKING":
            target_speed, stop_request, speed_reason = self.speed_planner.target_speed_for_event(
                current_speed_mps,
                desired_speed_mps,
                event.distance_m,
                event_type,
                True,
            )
            if stop_request or (event.distance_m is not None and event.distance_m <= float(self.get_parameter("stop_buffer_m").value)):
                return BehaviorState.PARKING, 0.0, True, speed_reason, "parking_stop"
            return BehaviorState.PARKING, target_speed, False, speed_reason, "parking_approach"

        if event_type == "RED_LIGHT":
            self.blocking_tl_active = True
            stop_required = event.distance_m is not None and event.distance_m <= float(self.get_parameter("stop_buffer_m").value)
            if stop_required:
                state = BehaviorState.TL_WAIT_GREEN if current_speed_mps <= 0.2 else BehaviorState.TL_STOP
                return state, 0.0, True, "red_light_hold", "red_light_stop"
            target_speed, _, speed_reason = self.speed_planner.target_speed_for_event(
                current_speed_mps,
                desired_speed_mps,
                event.distance_m,
                event_type,
                False,
            )
            return BehaviorState.TL_APPROACH, target_speed, False, speed_reason, "red_light_approach"

        if event_type == "YELLOW_LIGHT_STOP":
            if self._yellow_requires_stop(event, current_speed_mps):
                self.blocking_tl_active = True
                effective_event_type = "RED_LIGHT"
                stop_required = event.distance_m is not None and event.distance_m <= float(self.get_parameter("stop_buffer_m").value)
                if stop_required:
                    state = BehaviorState.TL_WAIT_GREEN if current_speed_mps <= 0.2 else BehaviorState.TL_STOP
                    return state, 0.0, True, "yellow_light_hold", "yellow_light_stop"
                target_speed, _, speed_reason = self.speed_planner.target_speed_for_event(
                    current_speed_mps,
                    desired_speed_mps,
                    event.distance_m,
                    effective_event_type,
                    False,
                )
                return BehaviorState.TL_APPROACH, target_speed, False, speed_reason, "yellow_light_stop_decision"
            self.blocking_tl_active = False
            target_speed, _, speed_reason = self.speed_planner.target_speed_for_event(
                current_speed_mps,
                desired_speed_mps,
                None,
                "NONE",
                False,
            )
            return BehaviorState.NORMAL_DRIVE, min(target_speed, desired_speed_mps), False, speed_reason, "yellow_light_pass_allowed"

        if (self.tl_event_payload or {}).get("color") == "green" and self.blocking_tl_active:
            self.blocking_tl_active = False
            return BehaviorState.TL_RELEASE, desired_speed_mps, False, "green_release", "green_light_release"

        if event_type in {"LEAD_VEHICLE", "LEAD_VEHICLE_STOPPED"}:
            target_speed, stop_request, speed_reason = self.speed_planner.target_speed_for_event(
                current_speed_mps,
                desired_speed_mps,
                event.distance_m,
                "LEAD_VEHICLE",
                event_type == "LEAD_VEHICLE_STOPPED",
                lead_vehicle_speed_mps=0.0,
                lead_vehicle_distance_m=event.distance_m,
            )
            return BehaviorState.NORMAL_DRIVE, 0.0 if stop_request else target_speed, stop_request, speed_reason, supervisor_reason

        self.blocking_tl_active = False
        target_speed, _, speed_reason = self.speed_planner.target_speed_for_event(
            current_speed_mps,
            desired_speed_mps,
            None,
            "NONE",
            False,
        )
        return BehaviorState.NORMAL_DRIVE, min(desired_speed_mps, target_speed), False, speed_reason, "no_blocking_event"

    def tick(self):
        now = time.time()
        lane_plan_timeout_s = float(self.get_parameter("lane_plan_timeout_s").value)
        status_timeout_s = float(self.get_parameter("status_timeout_s").value)
        if self.lane_plan_raw_payload is None or now - self.last_lane_plan_raw_s > lane_plan_timeout_s:
            payload = self._stop_output("missing_lane_plan_raw")
            self.publish_payload(payload)
            return
        if self.status_payload is None or now - self.last_status_s > status_timeout_s:
            payload = self._stop_output("missing_status")
            self.publish_payload(payload)
            return

        raw_plan = dict(self.lane_plan_raw_payload)
        desired_speed_mps = max(0.0, float(raw_plan.get("target_speed_mps", 0.0) or 0.0))
        current_speed_mps = self._current_speed_mps()
        route_points = list((self.route_payload or {}).get("points", []))

        event = self.event_finder.find_event(
            ego_status=self.status_payload or {},
            route_points=route_points,
            mission_payload=self.mission_payload,
            tl_event=self.tl_event_payload,
            lead_vehicle=None,
        )

        if bool((self.mission_payload or {}).get("must_stop", False)):
            event = RouteEvent(
                event_type="MISSION_STOP",
                distance_m=float((self.mission_payload or {}).get("distance_to_objective_m", 0.0) or 0.0),
                stop_point=None,
                color="mission",
                source="mission_node",
                priority=0,
                reason="mission_must_stop",
            )

        behavior_state, target_speed_mps, stop_request, speed_profile_reason, supervisor_reason = self._apply_behavior(
            event,
            current_speed_mps,
            desired_speed_mps,
        )

        self.state = behavior_state
        if stop_request:
            target_speed_mps = 0.0

        output = dict(raw_plan)
        output.update({
            "stamp": now,
            "source": "behavior_supervisor_node",
            "behavior_state": behavior_state.name,
            "event_type": event.event_type,
            "event_distance_m": round(float(event.distance_m), 3) if event.distance_m is not None else None,
            "event_source": event.source,
            "event_color": event.color,
            "speed_profile_reason": speed_profile_reason,
            "supervisor_reason": supervisor_reason,
            "raw_target_speed_mps": round(desired_speed_mps, 3),
            "target_speed_mps": round(float(target_speed_mps), 3),
            "stop_request": bool(stop_request),
            "reason": supervisor_reason if stop_request else speed_profile_reason,
            "stop_reason": supervisor_reason if stop_request else "",
        })
        output.setdefault("target_point", raw_plan.get("target_point") or self._fallback_target_point())
        self.publish_payload(output)

    def publish_payload(self, payload: dict):
        msg = String()
        msg.data = json.dumps(payload, ensure_ascii=False)
        self.plan_pub.publish(msg)
        self.runtime_logger.write({
            "behavior_state": payload.get("behavior_state"),
            "event_type": payload.get("event_type"),
            "event_distance_m": payload.get("event_distance_m"),
            "event_source": payload.get("event_source"),
            "event_color": payload.get("event_color"),
            "raw_target_speed_mps": payload.get("raw_target_speed_mps"),
            "target_speed_mps": payload.get("target_speed_mps"),
            "stop_request": payload.get("stop_request"),
            "speed_profile_reason": payload.get("speed_profile_reason"),
            "supervisor_reason": payload.get("supervisor_reason"),
        })
        now = time.time()
        period = float(self.get_parameter("ros_log_period_s").value)
        if now - self.last_ros_log_s >= period:
            self.last_ros_log_s = now
            self.get_logger().info(
                "behavior_supervisor "
                f"state={payload.get('behavior_state')} "
                f"event={payload.get('event_type')} "
                f"dist={payload.get('event_distance_m')} "
                f"target_v={payload.get('target_speed_mps')} "
                f"stop={payload.get('stop_request')} "
                f"reason={payload.get('supervisor_reason') or payload.get('speed_profile_reason')}"
            )


def main(args=None):
    rclpy.init(args=args)
    node = BehaviorSupervisorNode()
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
