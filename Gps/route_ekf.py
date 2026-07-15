"""EKF dẫn đường tầng 2: fusion encoder + GPS trên frame tuyến CSV.

Phân vai 2 tầng (xem thảo luận thiết kế):
  - Tầng 1 (control/ekf.py, FrenetEKF): bám làn khi CÓ vision — camera correct
    d/psi, encoder predict. Không đụng gì tới file này.
  - Tầng 2 (file này, RouteEKF): dẫn đường toàn tuyến — chạy LIÊN TỤC nền,
    encoder predict pose (x, y, psi) trong frame local của tuyến, GPS correct
    vị trí với R TRUNG THỰC (= hAcc^2). Nhờ R trung thực, phép Kalman tự cân:
    qua cua ngắn P còn nhỏ -> gain GPS ~ 0 (encoder thắng, GPS không giật xe);
    đoạn mù dài P phình -> GPS mới dần kéo lại chống trôi.

Dùng lúc CUA MẤT VISION: khi vào cua, anchor_lateral() chốt lệch ngang/heading
từ lần vision tốt cuối; trong cua route_state() trả (s, d, psi_err) so với
hình học tuyến CSV (chính xác tuyệt đối) để pure pursuit bám tiếp. Ra cua,
camera thấy line lại thì tầng 1 tự re-anchor, tầng 2 quay về vai trò nền.

Quy ước dấu tại biên (khớp "panel" trong control/ekf.py):
  - d > 0  = xe lệch sang PHẢI so với chiều đi của tuyến.
  - psi_err > 0 = mũi xe xoay sang PHẢI so với tiếp tuyến tuyến
    (để d_dot = v*sin(psi_err) đúng dấu, giống FrenetEKF).
  Nội bộ state psi là góc world CCW+ chuẩn ROS (psi += omega_odom*dt);
  đổi dấu chỉ diễn ra ở route_state()/anchor_lateral().

Thuần Python/numpy, không import rclpy — test offline được trước khi wiring.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

try:
    from .map_matcher import RouteMapMatcher  # dùng qua package Gps (vd Gps/node.py)
except ImportError:
    from map_matcher import RouteMapMatcher  # chạy truc tiep (vd python sim_route_ekf.py)


def _wrap(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


@dataclass(frozen=True)
class RouteState:
    """Pose hiện tại chiếu lên tuyến — đầu vào cho pure pursuit lúc mất vision."""

    s_m: float          # vị trí dọc tuyến [m]
    d_m: float          # lệch ngang, + = phải (quy ước panel) [m]
    psi_err: float      # lệch heading so với tiếp tuyến tuyến, + = phải [rad]
    progress: float     # s_m / tổng chiều dài tuyến
    sigma_d: float      # độ lệch chuẩn ước lượng của d [m] — trong cua mù P
                        # phình dần; consumer nên GIẢM TỐC khi sigma_d lớn
                        # (vd > 0.5 m) thay vì tin d mù quáng


@dataclass
class RouteEKFConfig:
    # Tune bằng sim_route_ekf.py (sweep 20 seed trên tuyến thật): q_xy=0.01/
    # q_psi=2e-4 cho max|d| qua cua mù nhỏ nhất; thấp hơn nữa EKF quá tự tin
    # (GPS không sửa nổi bias encoder), cao hơn thì nhiễu GPS lọt vào d.
    q_xy: float = 0.01          # process noise vị trí [m^2/s] (trôi encoder)
    q_psi: float = 2.0e-4       # process noise heading [rad^2/s] (trượt bánh khi cua)
    min_fix_type: int = 3       # GPS: chỉ nhận 3D fix trở lên
    max_h_acc_m: float = 5.0    # GPS: hAcc phải dưới ngưỡng
    default_h_acc_m: float = 3.0  # GPS không báo hAcc -> giả định mức này
    gps_gate_chi2: float = 9.21  # gate Mahalanobis 2 bậc tự do, 99% (chặn multipath)
    anchor_sigma_d: float = 0.15  # độ tin d từ vision lúc neo vào cua [m]
    anchor_sigma_psi: float = math.radians(3.0)  # độ tin heading lúc neo [rad]
    project_window_m: float = 40.0  # cửa sổ chiếu quanh s hiện tại (tránh match nhầm đoạn khác)
    # Trong curve zone (xem RouteEKF.curve_zones): cung cua ngắn (~10-20m ở
    # tuyến test) nên encoder dead-reckon rất chính xác, còn GPS dễ multipath/
    # che khuất hơn đúng lúc rẽ -> nhân thêm R ở đây để ưu tiên tin encoder
    # (predict) hơn GPS (correct) khi đang trong zone. 1.0 = tắt (R trung thực
    # như bình thường), càng lớn càng ít tin GPS trong cua.
    curve_gps_r_scale: float = 4.0


class RouteEKF:
    """State [x, y, psi] frame local tuyến; encoder predict, GPS correct."""

    def __init__(
        self,
        matcher: RouteMapMatcher,
        config: RouteEKFConfig | None = None,
        curve_zones: list[tuple[float, float]] | None = None,
    ) -> None:
        self.matcher = matcher
        self.config = config or RouteEKFConfig()
        # Cùng danh sách zone mà GpsNode dùng để gate curvature feed-forward
        # (xem Gps/node.py: matcher.detect_curve_zones(), curve_zone_* dùng
        # chung ở khối /** trong yaml) — truyền vào đây để correct_gps() biết
        # lúc nào nên bớt tin GPS.
        self.curve_zones = curve_zones or []
        self.x = np.zeros(3)
        self.P = np.diag([1e6, 1e6, 1e6])
        self.initialized = False
        self._s_hint: float | None = None  # s gần nhất, làm cửa sổ cho project_xy

    # ------------------------------------------------------------------ #

    def initialize(self, s_m: float, d_m: float = 0.0, psi_err: float = 0.0) -> None:
        """Đặt pose ban đầu từ (s, d, psi_err) trên tuyến (d/psi_err quy ước panel)."""
        rx, ry = self.matcher.point_at(s_m)
        theta = self.matcher.heading_at(s_m)
        # +d = phải = hướng pháp tuyến phải (sin(theta), -cos(theta)).
        self.x[0] = rx + d_m * math.sin(theta)
        self.x[1] = ry - d_m * math.cos(theta)
        # psi_err panel (+ = phải = cùng chiều kim đồng hồ) -> world CCW+: trừ.
        self.x[2] = _wrap(theta - psi_err)
        self.P = np.diag(
            [
                self.config.anchor_sigma_d**2 * 4.0,  # dọc tuyến kém tin hơn ngang một chút
                self.config.anchor_sigma_d**2 * 4.0,
                self.config.anchor_sigma_psi**2,
            ]
        )
        self._s_hint = float(s_m)
        self.initialized = True

    def predict(self, v_odom: float, omega_odom: float, dt: float) -> None:
        """Dead-reckoning từ encoder: v [m/s], omega [rad/s, CCW+ chuẩn ROS]."""
        if not self.initialized or dt <= 0.0:
            return
        psi = self.x[2]
        self.x[0] += v_odom * math.cos(psi) * dt
        self.x[1] += v_odom * math.sin(psi) * dt
        self.x[2] = _wrap(psi + omega_odom * dt)

        F = np.array(
            [
                [1.0, 0.0, -v_odom * math.sin(psi) * dt],
                [0.0, 1.0, v_odom * math.cos(psi) * dt],
                [0.0, 0.0, 1.0],
            ]
        )
        Q = np.diag([self.config.q_xy, self.config.q_xy, self.config.q_psi]) * dt
        self.P = F @ self.P @ F.T + Q
        self.P = 0.5 * (self.P + self.P.T)

    def _in_curve_zone(self) -> bool:
        """True nếu s ước lượng gần nhất (_s_hint) rơi vào 1 curve zone."""
        if self._s_hint is None or not self.curve_zones:
            return False
        return any(a <= self._s_hint <= b for a, b in self.curve_zones)

    def correct_gps(
        self,
        lat: float,
        lon: float,
        h_acc_m: float | None = None,
        fix_type: int | None = None,
    ) -> tuple[bool, str]:
        """Correct vị trí bằng fix GPS với R trung thực (= hAcc^2), nhân thêm
        curve_gps_r_scale nếu đang trong curve zone (xem _in_curve_zone) —
        cùng 1 fix GPS sẽ kéo state ít hơn hẳn khi đang ở khúc cua.

        Trả (accepted, reason). Fix rớt gate không làm hỏng state — bỏ qua thôi.
        """
        if not self.initialized:
            return False, "not-initialized"
        if fix_type is not None and fix_type < self.config.min_fix_type:
            return False, f"fix_type={fix_type} < {self.config.min_fix_type}"
        if h_acc_m is not None and h_acc_m > self.config.max_h_acc_m:
            return False, f"hAcc={h_acc_m:.1f}m > {self.config.max_h_acc_m}m"

        sigma = self.config.default_h_acc_m if h_acc_m is None else max(0.5, h_acc_m)
        R = np.diag([sigma**2, sigma**2])
        if self._in_curve_zone():
            R = R * self.config.curve_gps_r_scale
        H = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

        zx, zy = self.matcher.to_local(lat, lon)
        innov = np.array([zx - self.x[0], zy - self.x[1]])
        S = H @ self.P @ H.T + R

        # Gate Mahalanobis: innovation phải khả thi với P+R hiện tại (chặn multipath).
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return False, "S singular"
        maha = float(innov @ S_inv @ innov)
        if maha > self.config.gps_gate_chi2:
            return False, f"mahalanobis={maha:.1f} > {self.config.gps_gate_chi2}"

        K = self.P @ H.T @ S_inv
        self.x = self.x + K @ innov
        self.x[2] = _wrap(self.x[2])
        I_KH = np.eye(3) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T  # dạng Joseph, ổn định số
        self.P = 0.5 * (self.P + self.P.T)
        return True, "ok"

    def anchor_lateral(self, d_m: float, psi_err: float) -> None:
        """Chốt lệch ngang + heading từ vision (gọi lúc VÀO cua, khi camera còn line).

        Giữ nguyên thành phần dọc tuyến (s) — chỉ vision mới đủ tin để sửa ngang.
        d_m/psi_err theo quy ước panel như route_state().
        """
        if not self.initialized or self._s_hint is None:
            return
        state = self.route_state()
        s = state.s_m
        rx, ry = self.matcher.point_at(s)
        theta = self.matcher.heading_at(s)
        self.x[0] = rx + d_m * math.sin(theta)
        self.x[1] = ry - d_m * math.cos(theta)
        self.x[2] = _wrap(theta - psi_err)

        # Hiệp phương sai: giữ độ bất định DỌC tuyến hiện có, thu nhỏ NGANG +
        # heading về mức tin của vision. Xoay (along, cross) -> (x, y).
        t_vec = np.array([math.cos(theta), math.sin(theta)])
        sigma_along_sq = float(t_vec @ self.P[:2, :2] @ t_vec)
        rot = np.array(
            [[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]]
        )
        P_xy = rot @ np.diag([sigma_along_sq, self.config.anchor_sigma_d**2]) @ rot.T
        self.P = np.zeros((3, 3))
        self.P[:2, :2] = P_xy
        self.P[2, 2] = self.config.anchor_sigma_psi**2

    def route_state(self) -> RouteState:
        """Chiếu pose hiện tại lên tuyến -> (s, d, psi_err) cho pure pursuit."""
        if not self.initialized:
            raise RuntimeError("RouteEKF chua initialize()")
        if self._s_hint is None:
            s_lo, s_hi = None, None
        else:
            s_lo = self._s_hint - self.config.project_window_m
            s_hi = self._s_hint + self.config.project_window_m
        s_m, d_m, _seg = self.matcher.project_xy(self.x[0], self.x[1], s_lo, s_hi)
        self._s_hint = s_m
        theta = self.matcher.heading_at(s_m)
        psi_err = _wrap(theta - self.x[2])  # panel: + = mũi xe lệch phải
        # Chiếu P lên pháp tuyến phải của tuyến -> phương sai của d.
        n_vec = np.array([math.sin(theta), -math.cos(theta)])
        sigma_d = math.sqrt(max(0.0, float(n_vec @ self.P[:2, :2] @ n_vec)))
        return RouteState(
            s_m=s_m,
            d_m=d_m,
            psi_err=psi_err,
            progress=s_m / self.matcher.total_length_m,
            sigma_d=sigma_d,
        )
