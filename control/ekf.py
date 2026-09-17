"""EKF fusion giữa odometry encoder và quan sát làn từ perception.

State x = [d, psi]. Quy ước dấu của d/psi khớp với
/perception/frenet/optimal_path (xem planner_motion/logic.py: c_d = -d_meters),
NGƯỢC dấu với /perception/frenet/d. Lý do: pure_pursuit so sánh trực tiếp
d (EKF) với path_d (optimal_path) mà không cần đổi dấu ở mỗi tick — chỉ cần
đổi dấu 1 lần duy nhất khi nhận measurement từ perception.

Thuần Python/numpy, không import rclpy.
"""
from __future__ import annotations

import math

import numpy as np
from dataclasses import dataclass


def _wrap(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


@dataclass
class EKFState:
    d: float
    psi: float


class FrenetEKF:
    def __init__(
        self,
        q_d: float = 0.01,
        q_psi: float = 0.01,
        r_d: float = 0.04,
        r_psi: float = math.radians(5.0) ** 2,
    ) -> None:
        self.x = np.zeros(2)  # [d, psi]
        self.P = np.diag([0.5, 0.5])
        self.Q = np.diag([q_d, q_psi])
        self.R = np.diag([r_d, r_psi])

    def predict(self, v_odom: float, omega_odom: float, dt: float) -> None:
        if dt <= 0.0:
            return
        d, psi = self.x

        # psi (panel, +d = phải) = -psi_ROS (ROS: +y = trái) -> psi_dot =
        # -omega_odom, NGƯỢC dấu với tích phân yaw chuẩn ROS (REP103:
        # omega dương = CCW = rẽ trái = psi_ROS tăng = psi (panel) giảm).
        # XÁC NHẬN BẰNG TEST TAY THẬT: rẽ phải thật (omega_odom<0 chuẩn
        # REP103) -> psi_ROS giảm -> panel_psi = -psi_ROS phải TĂNG (+=phải).
        # Dùng "+omega_odom" cho ra psi giảm lúc rẽ phải thật (hiển thị như
        # đang rẽ trái) — đúng triệu chứng đã quan sát. Dùng "-omega_odom".
        x_pred = np.array([
            d - v_odom * math.sin(psi) * dt,
            _wrap(psi + omega_odom * dt),
        ])

        F = np.array([
            [1.0, v_odom * math.cos(psi) * dt],
            [0.0, 1.0],
        ])

        self.x = x_pred
        self.P = F @ self.P @ F.T + self.Q * dt

    def correct(self, d_meters_filtered: float, heading_filtered_deg: float) -> None:
        """Cập nhật d/psi từ perception. d_meters_filtered bị đổi dấu để
        khớp quy ước path (+d = xe ở bên phải reference); heading giữ nguyên
        dấu (đã khớp sẵn với c_d_d = c_speed*sin(hdg) trong
        planner_motion/logic.py).
        """
        z = np.array([
            -float(d_meters_filtered),
            math.radians(float(heading_filtered_deg)),
        ])
        H = np.array([
            [1.0, 0.0],
            [0.0, 1.0],
        ])

        y = z - H @ self.x
        # y[1] la innovation cua psi (goc) -> phai wrap ve (-pi, pi] truoc khi
        # dung, khong thi psi da troi qua bien +-pi (dead-reckon lau khong
        # correction, vd suot curve mode) se tao innovation khong lo gia tao
        # (vd 3.4 - (-0.2) = 3.6 rad du 2 goc vat ly gan nhu trung nhau) ->
        # Kalman gain nhan vao lam psi "nhay" dot ngot ngay lan correct ke tiep.
        y[1] = _wrap(y[1])
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.x[1] = _wrap(self.x[1])
        self.P = (np.eye(2) - K @ H) @ self.P

    def reset_lateral(self, d: float, psi: float) -> None:
        """Đặt lại d/psi (panel convention) — gọi lúc BÀN GIAO từ curve mode
        về vision (control/node.py:_planner_tick): suốt curve zone vision bị
        chặn không correct nên d/psi ở đây đã dead-reckon mù cả đoạn cua, vô
        nghĩa so với làn hiện tại; nạp giá trị cuối từ RouteEKF để vision
        correct hội tụ từ điểm hợp lý thay vì kéo từ giá trị trôi xa. P nới
        lên mức vừa phải để vài correction đầu sau đó ăn mạnh."""
        self.x[0] = float(d)
        self.x[1] = _wrap(float(psi))
        self.P[0, 0] = 0.25
        self.P[1, 1] = 0.25

    @property
    def state(self) -> EKFState:
        return EKFState(*self.x.tolist())
