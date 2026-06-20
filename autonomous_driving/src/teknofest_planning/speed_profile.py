from __future__ import annotations

import math


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def calc_distance(initial_speed_mps: float, final_speed_mps: float, accel_mps2: float) -> float:
    if abs(accel_mps2) < 1e-6:
        return 0.0
    distance = (final_speed_mps**2 - initial_speed_mps**2) / (2.0 * accel_mps2)
    return max(0.0, float(distance))


def calc_final_speed(initial_speed_mps: float, accel_mps2: float, distance_m: float) -> float:
    term = initial_speed_mps**2 + 2.0 * accel_mps2 * max(0.0, float(distance_m))
    if term <= 0.0:
        return 0.0
    return math.sqrt(term)


class SpeedProfilePlanner:
    def __init__(
        self,
        max_accel_mps2: float = 1.1,
        max_decel_mps2: float = 1.4,
        slow_speed_mps: float = 0.8,
        stop_buffer_m: float = 1.0,
        follow_time_gap_s: float = 1.4,
    ):
        self.max_accel_mps2 = max(0.1, float(max_accel_mps2))
        self.max_decel_mps2 = max(0.1, float(max_decel_mps2))
        self.slow_speed_mps = max(0.0, float(slow_speed_mps))
        self.stop_buffer_m = max(0.0, float(stop_buffer_m))
        self.follow_time_gap_s = max(0.5, float(follow_time_gap_s))
        self._last_target_speed_mps = 0.0

    def decelerate_profile(
        self,
        current_speed_mps: float,
        desired_speed_mps: float,
        distance_to_event_m: float | None,
    ) -> tuple[float, bool, str]:
        current_speed = max(0.0, float(current_speed_mps))
        desired_speed = max(0.0, float(desired_speed_mps))
        if distance_to_event_m is None:
            target = min(desired_speed, max(self.slow_speed_mps, current_speed * 0.5))
            return target, False, "event_distance_unknown_crawl"

        distance = max(0.0, float(distance_to_event_m))
        if distance <= self.stop_buffer_m:
            return 0.0, True, "event_within_stop_buffer"

        usable_distance = max(0.0, distance - self.stop_buffer_m)
        brake_to_stop_distance = calc_distance(current_speed, 0.0, -self.max_decel_mps2)
        brake_to_slow_distance = calc_distance(current_speed, self.slow_speed_mps, -self.max_decel_mps2)

        if usable_distance <= brake_to_stop_distance:
            target = calc_final_speed(0.0, self.max_decel_mps2, usable_distance)
            target = min(target, current_speed, desired_speed)
            return max(0.0, target), False, "decelerate_to_stop"

        if usable_distance <= brake_to_slow_distance:
            target = calc_final_speed(self.slow_speed_mps, self.max_decel_mps2, usable_distance)
            target = min(target, current_speed, desired_speed)
            return max(self.slow_speed_mps, target), False, "coast_to_slow_speed"

        ratio = clamp(usable_distance / max(brake_to_stop_distance + 1.0, 1.0), 0.0, 1.0)
        target = self.slow_speed_mps + (desired_speed - self.slow_speed_mps) * ratio
        return min(desired_speed, max(self.slow_speed_mps, target)), False, "approach_event"

    def follow_profile(
        self,
        current_speed_mps: float,
        desired_speed_mps: float,
        lead_vehicle_speed_mps: float | None,
        lead_vehicle_distance_m: float | None,
    ) -> tuple[float, bool, str]:
        current_speed = max(0.0, float(current_speed_mps))
        desired_speed = max(0.0, float(desired_speed_mps))
        lead_speed = max(0.0, float(lead_vehicle_speed_mps or 0.0))
        if lead_vehicle_distance_m is None:
            return min(desired_speed, lead_speed if lead_speed > 0.0 else desired_speed), False, "lead_vehicle_distance_unknown"

        lead_distance = max(0.0, float(lead_vehicle_distance_m))
        desired_gap = max(self.stop_buffer_m + 1.0, current_speed * self.follow_time_gap_s)
        emergency_gap = max(self.stop_buffer_m, desired_gap * 0.35)

        if lead_distance <= emergency_gap:
            return 0.0, True, "lead_vehicle_emergency_stop"

        if lead_distance <= desired_gap:
            ratio = clamp((lead_distance - emergency_gap) / max(desired_gap - emergency_gap, 0.1), 0.0, 1.0)
            follow_cap = lead_speed + ratio * max(0.0, desired_speed - lead_speed)
            target = min(desired_speed, max(0.0, follow_cap))
            return target, False, "lead_vehicle_follow"

        target = min(desired_speed, max(lead_speed, current_speed))
        return target, False, "lead_vehicle_clear"

    def nominal_profile(self, current_speed_mps: float, desired_speed_mps: float) -> tuple[float, bool, str]:
        desired_speed = max(0.0, float(desired_speed_mps))
        return desired_speed, False, "nominal_cruise"

    def _smooth_target(self, target_speed_mps: float) -> float:
        target = max(0.0, float(target_speed_mps))
        previous = max(0.0, float(self._last_target_speed_mps))
        max_rise = self.max_accel_mps2 * 0.1
        max_drop = self.max_decel_mps2 * 0.1
        delta = target - previous
        if delta > max_rise:
            target = previous + max_rise
        elif delta < -max_drop:
            target = previous - max_drop
        self._last_target_speed_mps = max(0.0, target)
        return self._last_target_speed_mps

    def target_speed_for_event(
        self,
        current_speed_mps: float,
        desired_speed_mps: float,
        distance_to_event_m: float | None,
        event_type: str,
        stop_required: bool,
        lead_vehicle_speed_mps: float | None = None,
        lead_vehicle_distance_m: float | None = None,
    ) -> tuple[float, bool, str]:
        event_name = str(event_type or "NONE").upper()
        desired_speed = max(0.0, float(desired_speed_mps))
        current_speed = max(0.0, float(current_speed_mps))

        if stop_required or event_name in {"RED_LIGHT", "MISSION_STOP", "PARKING", "YELLOW_LIGHT_STOP"}:
            target_speed, stop_request, reason = self.decelerate_profile(
                current_speed,
                desired_speed,
                distance_to_event_m,
            )
        elif event_name == "LEAD_VEHICLE":
            target_speed, stop_request, reason = self.follow_profile(
                current_speed,
                desired_speed,
                lead_vehicle_speed_mps,
                lead_vehicle_distance_m,
            )
        else:
            target_speed, stop_request, reason = self.nominal_profile(
                current_speed,
                desired_speed,
            )

        if stop_request:
            self._last_target_speed_mps = 0.0
            return 0.0, True, reason

        target_speed = min(desired_speed, max(0.0, float(target_speed)))
        target_speed = self._smooth_target(target_speed)
        target_speed = min(desired_speed, max(0.0, target_speed))
        return target_speed, False, reason
