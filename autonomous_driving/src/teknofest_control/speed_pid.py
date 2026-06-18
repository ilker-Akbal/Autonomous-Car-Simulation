from __future__ import annotations

from dataclasses import dataclass


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


@dataclass
class SpeedPidConfig:
    kp: float = 0.32
    ki: float = 0.04
    kd: float = 0.03
    integral_limit: float = 8.0
    accel_limit_per_s: float = 0.45
    decel_limit_per_s: float = 0.70
    throttle_deadband: float = 0.04
    brake_deadband: float = 0.04


class SpeedPid:
    def __init__(self, config: SpeedPidConfig):
        self.config = config
        self.integral = 0.0
        self.previous_error = 0.0
        self.previous_output = 0.0
        self.initialized = False

    def reset(self):
        self.integral = 0.0
        self.previous_error = 0.0
        self.previous_output = 0.0
        self.initialized = False

    def reset_integral(self):
        self.integral = 0.0

    def step(self, target_speed_mps: float, current_speed_mps: float, dt: float) -> float:
        dt = max(1e-3, float(dt))
        error = float(target_speed_mps) - float(current_speed_mps)

        self.integral = clamp(
            self.integral + error * dt,
            -self.config.integral_limit,
            self.config.integral_limit,
        )

        derivative = 0.0
        if self.initialized:
            derivative = (error - self.previous_error) / dt

        if error > 0.2 and self.previous_output < -self.config.brake_deadband:
            self.reset_integral()
        elif error < -0.2 and self.previous_output > self.config.throttle_deadband:
            self.reset_integral()

        raw_output = (
            self.config.kp * error
            + self.config.ki * self.integral
            + self.config.kd * derivative
        )
        raw_output = clamp(raw_output, -1.0, 1.0)

        if self.initialized:
            delta = raw_output - self.previous_output
            if delta > 0.0:
                delta = min(delta, self.config.accel_limit_per_s * dt)
            else:
                delta = max(delta, -self.config.decel_limit_per_s * dt)
            output = self.previous_output + delta
        else:
            output = raw_output

        output = clamp(output, -1.0, 1.0)
        self.previous_error = error
        self.previous_output = output
        self.initialized = True
        return output

    def split_throttle_brake(self, accel_command: float) -> tuple[float, float]:
        if accel_command >= self.config.throttle_deadband:
            return clamp(accel_command, 0.0, 1.0), 0.0

        if accel_command <= -self.config.brake_deadband:
            return 0.0, clamp(-accel_command, 0.0, 1.0)

        return 0.0, 0.0
