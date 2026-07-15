"""EKF fusion giữa odometry encoder và quan sát làn từ perception.

State x = [s, d, psi, v]. Quy ước dấu của d/psi khớp với
/perception/frenet/optimal_path (xem planner_motion/logic.py: c_d = -d_meters),
NGƯỢC dấu với /perception/frenet/d. Lý do: pure_pursuit so sánh trực tiếp
d (EKF) với path_d (optimal_path) mà không cần đổi dấu ở mỗi tick — chỉ cần
đổi dấu 1 lần duy nhất khi nhận measurement từ perception.

s: quãng đường đã đi kể từ lần correction gần nhất (mốc s=0 của optimal_path
hiện tại), vì optimal_path được tính lại từ vị trí xe mỗi frame camera mới
-> không có frame toàn cục cố định để theo dõi (x, y) tuyệt đối.

Thuần Python/numpy, không import rclpy.
"""
from __future__ import annotations

import math

import numpy as np
from dataclasses import dataclass


@dataclass
class EKFState:
    s: float
    d: float
    psi: float
    v: float


class FrenetEKF:
    def __init__(
        self,
        q_s: float = 0.02,
        q_d: float = 0.01,
        q_psi: float = 0.01,
        q_v: float = 0.05,
        r_d: float = 0.04,
        r_psi: float = math.radians(5.0) ** 2,
        r_v: float = 0.02,
    ) -> None:
        self.x = np.zeros(4)  # [s, d, psi, v]
        self.P = np.diag([1.0, 0.5, 0.5, 0.5])
        self.Q = np.diag([q_s, q_d, q_psi, q_v])
        self.R = np.diag([r_d, r_psi])
        self.r_v = r_v

    def predict(self, v_odom: float, omega_odom: float, dt: float) -> None:
        if dt <= 0.0:
            return
        s, d, psi, v = self.x

        # psi (panel, +d = phải) = -psi_ROS (ROS: +y = trái) -> psi_dot =
        # -omega_odom, NGƯỢC dấu với tích phân yaw chuẩn ROS (REP103:
        # omega dương = CCW = rẽ trái = psi_ROS tăng = psi (panel) giảm).
        x_pred = np.array([
            s + v_odom * math.cos(psi) * dt,
            d + v_odom * math.sin(psi) * dt,
            psi + omega_odom * dt,
            v,
        ])

        F = np.array([
            [1.0, 0.0, -v_odom * math.sin(psi) * dt, 0.0],
            [0.0, 1.0,  v_odom * math.cos(psi) * dt, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])

        self.x = x_pred
        self.P = F @ self.P @ F.T + self.Q * dt

        # Pseudo-measurement: /odom linear.x quan sát trực tiếp state v.
        H_v = np.array([[0.0, 0.0, 0.0, 1.0]])
        y = v_odom - (H_v @ self.x)[0]
        S = (H_v @ self.P @ H_v.T)[0, 0] + self.r_v
        K = (self.P @ H_v.T) / S
        self.x = self.x + (K.flatten() * y)
        self.P = (np.eye(4) - K @ H_v) @ self.P

    def correct(self, d_meters_filtered: float, heading_filtered_deg: float) -> None:
        """Cập nhật d/psi/v từ perception. d_meters_filtered bị đổi dấu để
        khớp quy ước path (+d = xe ở bên phải reference); heading giữ nguyên
        dấu (đã khớp sẵn với c_d_d = c_speed*sin(hdg) trong
        planner_motion/logic.py). KHÔNG đụng tới s — EKF chạy độc lập với
        perception, s chỉ reset khi planner replan (xem reset_s_origin()).
        """
        z = np.array([
            -float(d_meters_filtered),
            math.radians(float(heading_filtered_deg)),
        ])
        H = np.array([
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ])

        y = z - H @ self.x
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ H) @ self.P

    def reset_s_origin(self) -> None:
        """Mốc s=0 mới — gọi sau mỗi lần replan (planner đã tính path mới
        bắt đầu từ vị trí hiện tại của xe)."""
        self.x[0] = 0.0
        self.P[0, 0] = 1.0

    @property
    def state(self) -> EKFState:
        return EKFState(*self.x.tolist())
