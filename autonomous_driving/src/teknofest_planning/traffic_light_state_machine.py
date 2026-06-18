from __future__ import annotations

import math
import time
from dataclasses import dataclass


NO_RELEVANT_LIGHT = "NO_RELEVANT_LIGHT"
RED_TRACKED = "RED_TRACKED"
RED_DECEL = "RED_DECEL"
RED_STOP_COMMIT = "RED_STOP_COMMIT"
RED_STOP_CREEP = "RED_STOP_CREEP"
RED_LOCK_OVERRUN_HOLD = "RED_LOCK_OVERRUN_HOLD"
STOPPED_AT_RED = "STOPPED_AT_RED"
GREEN_RELEASE = "GREEN_RELEASE"
YELLOW_APPROACH = "YELLOW_APPROACH"
YELLOW_CAN_STOP = "YELLOW_CAN_STOP"
YELLOW_POINT_OF_NO_RETURN = "YELLOW_POINT_OF_NO_RETURN"
PASSED_LIGHT = "PASSED_LIGHT"
POST_LIGHT_IGNORE = "POST_LIGHT_IGNORE"


@dataclass
class TrafficLightDecision:
    state: str
    desired_speed_mps: float
    desired_speed_raw_mps: float
    stop_request: bool
    required_stop_distance_m: float
    point_of_no_return: bool
    passed_light: bool
    post_light_ignore_active: bool
    reason: str
    red_stop_commit_threshold_m: float = 0.0
    red_stop_commit_active: bool = False
    stop_commit_reason: str = ""
    red_creep_active: bool = False
    red_creep_target_mps: float = 0.0
    red_creep_remaining_m: float = 0.0
    red_creep_elapsed_s: float = 0.0
    red_creep_stop_threshold_m: float = 0.0
    red_creep_reason: str = ""
    stopped_too_far_from_stop_point: bool = False
    final_stop_distance_m: float = 0.0
    visual_red_brake: bool = False
    visual_red_commit: bool = False
    visual_red_hard_commit: bool = False
    visual_red_approach: bool = False
    visual_red_approach_target_mps: float = 0.0


class TrafficLightStateMachine:
    def __init__(
        self,
        *,
        comfort_decel: float = 1.4,
        max_decel: float = 3.0,
        safe_stop_buffer: float = 1.8,
        reaction_margin: float = 1.5,
        post_light_ignore_s: float = 3.0,
        speed_slew_accel_mps2: float = 1.2,
        speed_slew_decel_mps2: float = 1.4,
        stop_commit_distance_m: float = 1.2,
        full_stop_speed_mps: float = 0.25,
        red_creep_min_distance_m: float = 0.7,
        red_creep_max_distance_m: float = 6.0,
        red_creep_speed_mps: float = 0.45,
        red_creep_max_s: float = 4.0,
        target_red_stop_distance_m: float = 2.0,
        red_stop_commit_margin_m: float = 0.25,
    ):
        self.comfort_decel = max(0.1, float(comfort_decel))
        self.max_decel = max(self.comfort_decel, float(max_decel))
        self.safe_stop_buffer = max(0.0, float(safe_stop_buffer))
        self.reaction_margin = max(0.0, float(reaction_margin))
        self.post_light_ignore_s = max(0.0, float(post_light_ignore_s))
        self.speed_slew_accel_mps2 = max(0.1, float(speed_slew_accel_mps2))
        self.speed_slew_decel_mps2 = max(0.1, float(speed_slew_decel_mps2))
        self.stop_commit_distance_m = max(0.0, float(stop_commit_distance_m))
        self.full_stop_speed_mps = max(0.0, float(full_stop_speed_mps))
        self.red_creep_min_distance_m = max(0.0, float(red_creep_min_distance_m))
        self.red_creep_max_distance_m = max(self.red_creep_min_distance_m, float(red_creep_max_distance_m))
        self.red_creep_speed_mps = max(0.0, float(red_creep_speed_mps))
        self.red_creep_max_s = max(0.0, float(red_creep_max_s))
        self.target_red_stop_distance_m = max(0.0, float(target_red_stop_distance_m))
        self.final_stop_distance_m = self.target_red_stop_distance_m
        self.red_stop_commit_margin_m = max(0.0, float(red_stop_commit_margin_m))

        self.state = NO_RELEVANT_LIGHT
        self.smoothed_speed_mps = None
        self.ignore_until_s = 0.0
        self.red_creep_started_s = None
        self.last_tick_s = time.time()

    def reset_speed(self, speed_mps: float):
        self.smoothed_speed_mps = max(0.0, float(speed_mps))

    def speed_slew_limit(self, raw_speed: float, dt: float) -> float:
        raw_speed = max(0.0, float(raw_speed))
        if self.smoothed_speed_mps is None:
            self.smoothed_speed_mps = raw_speed
            return raw_speed

        delta = raw_speed - self.smoothed_speed_mps
        rate = self.speed_slew_accel_mps2 if delta >= 0.0 else self.speed_slew_decel_mps2
        max_delta = rate * max(1e-3, float(dt))
        if abs(delta) <= max_delta:
            self.smoothed_speed_mps = raw_speed
        else:
            self.smoothed_speed_mps += math.copysign(max_delta, delta)
        return self.smoothed_speed_mps

    def desired_stop_speed(self, remaining_distance_m: float, cruise_speed_mps: float) -> float:
        usable_distance = max(float(remaining_distance_m) - self.safe_stop_buffer, 0.0)
        desired = math.sqrt(max(0.0, 2.0 * self.comfort_decel * usable_distance))
        return min(float(cruise_speed_mps), desired)

    def desired_red_approach_speed(self, remaining_distance_m: float, cruise_speed_mps: float) -> float:
        usable_distance = max(float(remaining_distance_m) - self.target_red_stop_distance_m, 0.0)
        desired = math.sqrt(max(0.0, 2.0 * self.comfort_decel * usable_distance))
        return min(float(cruise_speed_mps), desired)

    def required_stop_distance(self, current_speed_mps: float) -> float:
        return (float(current_speed_mps) ** 2) / (2.0 * self.comfort_decel) + self.reaction_margin

    def update(
        self,
        *,
        relevant: bool,
        color: str,
        stop_point_distance_m: float | None,
        current_speed_mps: float,
        cruise_speed_mps: float,
        distance_valid: bool,
        visual_stopline_active: bool = False,
        visual_stopline_distance_m: float | None = None,
        visual_red_brake_distance_m: float = 6.0,
        visual_red_commit_distance_m: float = 2.5,
        visual_red_hard_commit_distance_m: float = 1.5,
        visual_red_brake_target_mps: float = 1.8,
        visual_red_approach_target_mps: float = 0.5,
    ) -> TrafficLightDecision:
        now = time.time()
        dt = max(1e-3, now - self.last_tick_s)
        self.last_tick_s = now

        cruise_speed = max(0.0, float(cruise_speed_mps))
        current_speed = max(0.0, float(current_speed_mps))
        remaining = float(stop_point_distance_m) if stop_point_distance_m is not None else None
        color = str(color or "unknown").lower()
        ignore_active = now < self.ignore_until_s
        passed = remaining is not None and remaining <= -1.0
        required_distance = self.required_stop_distance(current_speed)
        red_stop_commit_threshold = self.target_red_stop_distance_m + self.red_stop_commit_margin_m

        if color == "green" and relevant:
            self.red_creep_started_s = None
            self.state = GREEN_RELEASE
            desired = self.speed_slew_limit(cruise_speed, dt)
            return TrafficLightDecision(
                state=self.state,
                desired_speed_mps=desired,
                desired_speed_raw_mps=cruise_speed,
                stop_request=False,
                required_stop_distance_m=required_distance,
                point_of_no_return=False,
                passed_light=False,
                post_light_ignore_active=False,
                reason="green_release",
            )

        if ignore_active:
            self.state = POST_LIGHT_IGNORE
            desired = self.speed_slew_limit(cruise_speed, dt)
            return TrafficLightDecision(
                state=self.state,
                desired_speed_mps=desired,
                desired_speed_raw_mps=cruise_speed,
                stop_request=False,
                required_stop_distance_m=required_distance,
                point_of_no_return=False,
                passed_light=bool(passed),
                post_light_ignore_active=True,
                reason="post_light_ignore",
            )

        if passed and self.state not in {NO_RELEVANT_LIGHT, POST_LIGHT_IGNORE}:
            self.state = PASSED_LIGHT
            self.ignore_until_s = now + self.post_light_ignore_s
            desired = self.speed_slew_limit(cruise_speed, dt)
            return TrafficLightDecision(
                state=self.state,
                desired_speed_mps=desired,
                desired_speed_raw_mps=cruise_speed,
                stop_request=False,
                required_stop_distance_m=required_distance,
                point_of_no_return=False,
                passed_light=True,
                post_light_ignore_active=False,
                reason="passed_light",
            )

        if not relevant or not distance_valid or remaining is None:
            self.state = NO_RELEVANT_LIGHT
            self.reset_speed(cruise_speed)
            return TrafficLightDecision(
                state=self.state,
                desired_speed_mps=cruise_speed,
                desired_speed_raw_mps=cruise_speed,
                stop_request=False,
                required_stop_distance_m=required_distance,
                point_of_no_return=False,
                passed_light=False,
                post_light_ignore_active=False,
                reason="no_relevant_light",
            )

        if color == "green":
            self.red_creep_started_s = None
            self.state = GREEN_RELEASE
            desired = self.speed_slew_limit(cruise_speed, dt)
            return TrafficLightDecision(
                state=self.state,
                desired_speed_mps=desired,
                desired_speed_raw_mps=cruise_speed,
                stop_request=False,
                required_stop_distance_m=required_distance,
                point_of_no_return=False,
                passed_light=False,
                post_light_ignore_active=False,
                reason="green_release",
            )

        if color == "yellow":
            can_stop = remaining > required_distance
            if can_stop:
                raw = self.desired_stop_speed(remaining, cruise_speed)
                desired = self.speed_slew_limit(raw, dt)
                commit = remaining <= self.stop_commit_distance_m
                self.state = YELLOW_CAN_STOP if not commit else RED_STOP_COMMIT
                return TrafficLightDecision(
                    state=self.state,
                    desired_speed_mps=desired,
                    desired_speed_raw_mps=raw,
                    stop_request=bool(commit),
                    required_stop_distance_m=required_distance,
                    point_of_no_return=False,
                    passed_light=False,
                    post_light_ignore_active=False,
                    reason="yellow_can_stop",
                )

            self.state = YELLOW_POINT_OF_NO_RETURN
            desired = self.speed_slew_limit(cruise_speed, dt)
            return TrafficLightDecision(
                state=self.state,
                desired_speed_mps=desired,
                desired_speed_raw_mps=cruise_speed,
                stop_request=False,
                required_stop_distance_m=required_distance,
                point_of_no_return=True,
                passed_light=False,
                post_light_ignore_active=False,
                reason="yellow_point_of_no_return",
            )

        if color == "red":
            self.red_creep_started_s = None
            visual_remaining = (
                float(visual_stopline_distance_m)
                if visual_stopline_distance_m is not None
                else remaining
            )
            if visual_stopline_active and visual_remaining is not None:
                visual_red_brake = 0.0 < visual_remaining <= visual_red_brake_distance_m
                visual_red_commit = 0.0 < visual_remaining <= visual_red_commit_distance_m
                visual_red_hard_commit = 0.0 < visual_remaining <= visual_red_hard_commit_distance_m
                visual_red_approach = bool(visual_red_brake and not visual_red_commit)
                stop_commit_reason = ""

                if visual_red_commit:
                    raw = 0.0
                    desired = 0.0
                    stopped = current_speed <= self.full_stop_speed_mps
                    self.state = STOPPED_AT_RED if stopped else RED_STOP_COMMIT
                    stop_commit_reason = "red_stop_2m_visual_stopline"
                elif visual_red_approach:
                    raw = max(0.0, float(visual_red_approach_target_mps))
                    desired = raw
                    self.smoothed_speed_mps = raw
                    stopped = False
                    self.state = RED_DECEL
                else:
                    raw = cruise_speed
                    desired = self.speed_slew_limit(raw, dt)
                    stopped = False
                    self.state = RED_TRACKED

                return TrafficLightDecision(
                    state=self.state,
                    desired_speed_mps=desired,
                    desired_speed_raw_mps=raw,
                    stop_request=bool(visual_red_commit or stopped),
                    required_stop_distance_m=required_distance,
                    point_of_no_return=False,
                    passed_light=False,
                    post_light_ignore_active=False,
                    reason=(
                        "red_stop_2m_visual_stopline"
                        if visual_red_commit
                        else "red_visual_slow_10m_rule"
                        if visual_red_approach
                        else RED_TRACKED.lower()
                    ),
                    red_stop_commit_threshold_m=max(0.0, float(visual_red_commit_distance_m)),
                    red_stop_commit_active=bool(visual_red_commit),
                    stop_commit_reason=stop_commit_reason,
                    stopped_too_far_from_stop_point=False,
                    final_stop_distance_m=max(0.0, float(visual_red_commit_distance_m)),
                    visual_red_brake=bool(visual_red_brake),
                    visual_red_commit=bool(visual_red_commit),
                    visual_red_hard_commit=bool(visual_red_hard_commit),
                    visual_red_approach=bool(visual_red_approach),
                    visual_red_approach_target_mps=(
                        max(0.0, float(visual_red_approach_target_mps))
                        if visual_red_approach
                        else 0.0
                    ),
                )

            raw = self.desired_red_approach_speed(remaining, cruise_speed)
            visual_red_brake = bool(
                visual_stopline_active
                and visual_remaining is not None
                and 0.0 < visual_remaining <= visual_red_brake_distance_m
            )
            visual_red_commit = bool(
                visual_stopline_active
                and visual_remaining is not None
                and 0.0 < visual_remaining <= visual_red_commit_distance_m
            )
            visual_red_hard_commit = bool(
                visual_stopline_active
                and visual_remaining is not None
                and 0.0 < visual_remaining <= visual_red_hard_commit_distance_m
            )
            visual_red_approach = bool(
                visual_red_brake
                and not visual_red_commit
                and visual_remaining is not None
                and visual_remaining > visual_red_commit_distance_m + 0.5
            )
            if visual_red_brake and not visual_red_commit:
                target = (
                    visual_red_approach_target_mps
                    if visual_red_approach
                    else visual_red_brake_target_mps
                )
                raw = min(raw, max(0.0, float(target)))
            commit = remaining > 0.0 and (
                remaining <= red_stop_commit_threshold or visual_red_commit
            )
            stopped = commit and current_speed <= self.full_stop_speed_mps
            stop_commit_reason = ""

            desired = self.speed_slew_limit(raw, dt)
            if visual_red_approach:
                desired = raw
                self.smoothed_speed_mps = raw
            if stopped:
                self.state = STOPPED_AT_RED
                stop_commit_reason = (
                    "stopped_at_visual_red_stopline"
                    if visual_red_commit
                    else "stopped_at_red_target_distance"
                )
                desired = 0.0
                raw = 0.0
            elif commit:
                self.state = RED_STOP_COMMIT
                stop_commit_reason = (
                    "red_stop_at_visual_stopline"
                    if visual_red_commit
                    else "red_stop_at_target_distance"
                )
                desired = 0.0
                raw = 0.0
            elif desired < cruise_speed - 0.05:
                self.state = RED_DECEL
            else:
                self.state = RED_TRACKED

            return TrafficLightDecision(
                state=self.state,
                desired_speed_mps=desired,
                desired_speed_raw_mps=raw,
                stop_request=bool(commit or stopped),
                required_stop_distance_m=required_distance,
                point_of_no_return=False,
                passed_light=False,
                post_light_ignore_active=False,
                reason=(
                    "stopped_at_visual_red_stopline"
                    if stopped and visual_red_commit
                    else "stopped_at_red_target_distance"
                    if stopped
                    else "red_stop_at_visual_stopline"
                    if commit and visual_red_commit
                    else "red_stop_at_target_distance"
                    if commit
                    else "visual_red_approach_to_stopline"
                    if visual_red_approach
                    else self.state.lower()
                ),
                red_stop_commit_threshold_m=red_stop_commit_threshold,
                red_stop_commit_active=bool(commit),
                stop_commit_reason=stop_commit_reason,
                stopped_too_far_from_stop_point=False,
                final_stop_distance_m=self.target_red_stop_distance_m,
                visual_red_brake=visual_red_brake,
                visual_red_commit=visual_red_commit,
                visual_red_hard_commit=visual_red_hard_commit,
                visual_red_approach=visual_red_approach,
                visual_red_approach_target_mps=(
                    max(0.0, float(visual_red_approach_target_mps))
                    if visual_red_approach
                    else 0.0
                ),
            )

        self.state = NO_RELEVANT_LIGHT
        self.reset_speed(cruise_speed)
        return TrafficLightDecision(
            state=self.state,
            desired_speed_mps=cruise_speed,
            desired_speed_raw_mps=cruise_speed,
            stop_request=False,
            required_stop_distance_m=required_distance,
            point_of_no_return=False,
            passed_light=False,
            post_light_ignore_active=False,
            reason="unknown_color",
        )
