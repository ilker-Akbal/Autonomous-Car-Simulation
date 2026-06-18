from __future__ import annotations

import math
from dataclasses import dataclass


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


@dataclass
class VehiclePose:
    x: float
    y: float
    yaw_deg: float


@dataclass
class TargetPoint:
    x: float
    y: float


@dataclass
class PurePursuitConfig:
    wheel_base_m: float = 2.85
    max_steer_rad: float = 0.70
    steer_sign: float = 1.0
    max_steer_delta_per_s: float = 2.6
    steer_low_pass_alpha: float = 0.45


class PurePursuit:
    def __init__(self, config: PurePursuitConfig):
        self.config = config
        self.previous_steer = 0.0
        self.initialized = False
        self.last_steer_raw = 0.0
        self.last_steer_smoothed = 0.0
        self.last_steer_rate_limited = 0.0
        self.last_rate_limited = False

    def reset(self):
        self.previous_steer = 0.0
        self.initialized = False
        self.last_steer_raw = 0.0
        self.last_steer_smoothed = 0.0
        self.last_steer_rate_limited = 0.0
        self.last_rate_limited = False

    def force_previous_steer(self, steer: float):
        steer = clamp(float(steer), -1.0, 1.0)
        self.previous_steer = steer
        self.last_steer_rate_limited = steer
        self.initialized = True

    def compute(self, pose: VehiclePose, target: TargetPoint, dt: float) -> float:
        dx = float(target.x) - float(pose.x)
        dy = float(target.y) - float(pose.y)
        yaw = math.radians(float(pose.yaw_deg))

        local_x = math.cos(yaw) * dx + math.sin(yaw) * dy
        local_y = -math.sin(yaw) * dx + math.cos(yaw) * dy

        lookahead = max(1.0, math.hypot(local_x, local_y))
        curvature = (2.0 * local_y) / (lookahead * lookahead)
        steer_rad = math.atan(self.config.wheel_base_m * curvature)
        raw_steer = self.config.steer_sign * steer_rad / max(1e-3, self.config.max_steer_rad)
        raw_steer = clamp(raw_steer, -1.0, 1.0)

        alpha = clamp(float(self.config.steer_low_pass_alpha), 0.0, 1.0)
        if self.initialized:
            smoothed = self.previous_steer + alpha * (raw_steer - self.previous_steer)
        else:
            smoothed = raw_steer

        rate_limited = False
        steer = smoothed
        if self.initialized:
            max_delta = self.config.max_steer_delta_per_s * max(1e-3, dt)
            delta = clamp(steer - self.previous_steer, -max_delta, max_delta)
            rate_limited = abs(delta - (steer - self.previous_steer)) > 1e-6
            steer = self.previous_steer + delta

        self.last_steer_raw = raw_steer
        self.last_steer_smoothed = smoothed
        self.last_steer_rate_limited = steer
        self.last_rate_limited = rate_limited
        self.previous_steer = steer
        self.initialized = True
        return clamp(steer, -1.0, 1.0)
