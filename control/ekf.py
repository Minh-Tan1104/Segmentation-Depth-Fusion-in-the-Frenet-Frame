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


def _wrap(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


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
        # XÁC NHẬN BẰNG TEST TAY THẬT: rẽ phải thật (omega_odom<0 chuẩn
        # REP103) -> psi_ROS giảm -> panel_psi = -psi_ROS phải TĂNG (+=phải).
        # Dùng "+omega_odom" cho ra psi giảm lúc rẽ phải thật (hiển thị như
        # đang rẽ trái) — đúng triệu chứng đã quan sát. Dùng "-omega_odom".
        x_pred = np.array([
            s + v_odom * math.cos(psi) * dt,
            d - v_odom * math.sin(psi) * dt,
            _wrap(psi + omega_odom * dt),
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
        # y[1] la innovation cua psi (goc) -> phai wrap ve (-pi, pi] truoc khi
        # dung, khong thi psi da troi qua bien +-pi (dead-reckon lau khong
        # correction, vd suot curve mode) se tao innovation khong lo gia tao
        # (vd 3.4 - (-0.2) = 3.6 rad du 2 goc vat ly gan nhu trung nhau) ->
        # Kalman gain nhan vao lam psi "nhay" dot ngot ngay lan correct ke tiep.
        y[1] = _wrap(y[1])
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.x[2] = _wrap(self.x[2])
        self.P = (np.eye(4) - K @ H) @ self.P

    def reset_lateral(self, d: float, psi: float) -> None:
        """Đặt lại d/psi (panel convention) — gọi lúc BÀN GIAO từ curve mode
        về vision (control/node.py:_planner_tick): suốt curve zone vision bị
        chặn không correct nên d/psi ở đây đã dead-reckon mù cả đoạn cua, vô
        nghĩa so với làn hiện tại; nạp giá trị cuối từ RouteEKF để vision
        correct hội tụ từ điểm hợp lý thay vì kéo từ giá trị trôi xa. P nới
        lên mức vừa phải để vài correction đầu sau đó ăn mạnh."""
        self.x[1] = float(d)
        self.x[2] = _wrap(float(psi))
        self.P[1, 1] = 0.25
        self.P[2, 2] = 0.25

    def reset_s_origin(self) -> None:
        """Mốc s=0 mới — gọi sau mỗi lần replan (planner đã tính path mới
        bắt đầu từ vị trí hiện tại của xe)."""
        self.x[0] = 0.0
        self.P[0, 0] = 1.0

    @property
    def state(self) -> EKFState:
        return EKFState(*self.x.tolist())
