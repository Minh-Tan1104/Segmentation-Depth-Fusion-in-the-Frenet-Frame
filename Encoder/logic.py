"""Differential-drive odometry và mapping cmd_vel -> lệnh raw hoverboard.

Thuần Python/numpy, không import rclpy — để dễ test độc lập.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class OdomConfig:
    wheel_radius: float = 0.0762   # m
    track_width: float = 0.3556    # m
    right_wheel_sign: float = -1.0  # encoder bánh phải bị ngược dấu khi lắp đặt


class DifferentialOdometry:
    """Tích phân (x, y, theta) từ vận tốc bánh trái/phải (RPM)."""

    def __init__(self, config: OdomConfig) -> None:
        self.config = config
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.v = 0.0
        self.omega = 0.0

    def reset(self) -> None:
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.v = 0.0
        self.omega = 0.0

    def update(self, speedL_rpm: float, speedR_rpm: float, dt: float) -> tuple[float, float, float, float, float]:
        if dt <= 0.0:
            return self.x, self.y, self.theta, self.v, self.omega

        rpm_to_mps = 2.0 * math.pi * self.config.wheel_radius / 60.0
        vL = speedL_rpm * rpm_to_mps
        vR = (self.config.right_wheel_sign * speedR_rpm) * rpm_to_mps

        v = (vL + vR) / 2.0
        omega = (vR - vL) / self.config.track_width

        self.theta += omega * dt
        self.x += v * math.cos(self.theta) * dt
        self.y += v * math.sin(self.theta) * dt
        self.v = v
        self.omega = omega

        return self.x, self.y, self.theta, self.v, self.omega


@dataclass
class CmdVelToRawConfig:
    cmd_speed_scale: float = 25.0   # raw unit / (m/s)
    cmd_steer_scale: float = 75.0   # raw unit / (rad/s)
    max_speed_raw: int = 50         # MAX_SPEED firmware
    max_steer_raw: int = 150        # MAX_STEER firmware


def _clamp(value: float, limit: float) -> float:
    return max(-limit, min(limit, value))


def cmd_vel_to_raw(linear_x: float, angular_z: float, cfg: CmdVelToRawConfig) -> tuple[int, int]:
    """Quy đổi (linear.x [m/s], angular.z [rad/s]) -> (steer, speed) raw int16.

    speed_raw điều khiển tốc độ tiến/lùi, steer_raw là lệnh rẽ vi sai mà
    firmware hoverboard tự cộng/trừ cho 2 bánh. Hệ số quy đổi (cmd_speed_scale,
    cmd_steer_scale) là tham số ROS cần tinh chỉnh thực nghiệm trên xe thật.
    """
    speed_raw = int(round(_clamp(linear_x * cfg.cmd_speed_scale, cfg.max_speed_raw)))
    steer_raw = int(round(_clamp(angular_z * cfg.cmd_steer_scale, cfg.max_steer_raw)))
    return steer_raw, speed_raw
