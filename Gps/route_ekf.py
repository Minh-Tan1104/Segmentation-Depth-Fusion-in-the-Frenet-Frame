"""EKF dẫn đường tầng 2: fusion encoder + GPS trên frame tuyến CSV.

Phân vai 2 tầng (xem thảo luận thiết kế):
  - Tầng 1 (control/ekf.py, FrenetEKF): bám làn khi CÓ vision — camera correct
    d/psi, encoder predict. Không đụng gì tới file này.
  - Tầng 2 (file này, RouteEKF): dẫn đường toàn tuyến — chạy LIÊN TỤC nền,
    encoder predict pose (x, y, psi) trong frame local của tuyến, GPS correct
    vị trí với R TRUNG THỰC (= hAcc^2) khi NGOÀI curve zone. Trong curve zone
    (có hysteresis, xem _update_curve_zone_state()) GPS bị CHẶN HẲN
    (correct_gps() từ chối luôn) — cung cua ngắn nên encoder dead-reckon đủ
    chính xác, còn GPS dễ multipath/che khuất đúng lúc rẽ; ra khỏi zone GPS
    lại kéo về chống trôi như bình thường.

Dùng lúc CUA MẤT VISION: ĐÚNG tại cạnh lên của curve zone, anchor_lateral()
chốt lệch ngang/heading từ lần vision tốt cuối làm ĐIỀU KIỆN ĐẦU; suốt phần
còn lại của cua route_state() trả (s, d, psi_err) so với hình học tuyến CSV
(chính xác tuyệt đối) để pure pursuit bám tiếp, nguồn duy nhất là encoder.
Ra cua, camera thấy line lại thì tầng 1 tự re-anchor, tầng 2 quay về vai trò nền.

VISION KHÔNG BAO GIỜ correct state ngoài thời điểm neo đó — ngoài curve zone
tầng 2 là GPS + encoder THUẦN. Xem anchor_lateral() và Gps/node.py
(anchor_on_curve_entry_only) để biết vì sao gọi lặp mỗi tick là sai.

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
    # che khuất hơn đúng lúc rẽ -> CHẶN HẲN correct_gps() khi đang trong zone
    # (xem correct_gps()), chỉ còn predict() từ encoder quyết định pose qua
    # cua. curve_zone_hysteresis_m: khi ĐÃ ở trong zone, phải đi xa hơn biên
    # gốc [a,b] thêm mức này mới coi là RA khỏi zone (còn lúc CHƯA ở trong
    # zone thì vẫn dùng đúng biên gốc để vào) — tránh zone "nhấp nháy" on/off
    # mỗi tick khi s ước lượng dao động sát biên do nhiễu GPS/encoder.
    curve_zone_hysteresis_m: float = 2.0
    # true (mặc định) = CHẶN HẲN correct_gps() khi đang trong curve zone (như
    # cũ: chỉ tin encoder dead-reckon qua cua). false = VẪN CHO PHÉP GPS
    # correct_gps() trong curve zone (GPS vẫn phải qua gate min_fix_type,
    # max_h_acc_m, Mahalanobis bình thường).
    curve_zone_block_gps: bool = True


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
        self._in_zone_state = False  # trạng thái curve-zone hiện tại (có hysteresis)

    # ------------------------------------------------------------------ #

    def initialize(self, s_m: float, d_m: float = 0.0, psi_err: float = 0.0) -> None:
        """Đặt pose ban đầu từ (s, d, psi_err) trên tuyến (d/psi_err quy ước panel)."""
        rx, ry = self.matcher.point_at(s_m)
        theta = self.matcher.heading_at(s_m)
        # +d = phải = hướng pháp tuyến phải (sin(theta), -cos(theta)).
        self.x[0] = rx + d_m * math.sin(theta)
        self.x[1] = ry - d_m * math.cos(theta)
        # psi_err panel (+ = phải = cùng chiều kim đồng hồ) -> world CCW+: trừ.
        self.x[2] = _wrap(theta + psi_err)
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
        """Dead-reckoning từ encoder: v [m/s], omega [rad/s từ /odom angular.z].

        DẤU omega: state psi ở đây là heading WORLD chuẩn (frame local tuyến,
        CCW+ giống REP103 — psi tăng khi rẽ trái) — mô hình unicycle chuẩn,
        không quy đổi qua frame panel nào cả. Vì vậy CHỈ phụ thuộc đúng bản
        chất vật lý của omega_odom (CCW+), KHÔNG phụ thuộc FrenetEKF
        (control/ekf.py) đang dùng dấu gì cho psi PANEL của nó — 2 state này
        không bao giờ trừ/so trực tiếp với nhau ở bất kỳ đâu (route_state().
        psi_err chỉ dùng nội bộ RouteEKF: theta_tuyến − self.x[2]; đường cũ ở
        control/node.py._planner_tick nhận psi_err này qua /gps/route_state
        nhưng BỎ HẲN không dùng — psi_for_plan luôn lấy từ FrenetEKF.state.psi).
        Đã verify bằng mô phỏng: cho omega_odom = tốc độ góc THẬT dọc 1 khúc
        cua thật trên map/gps_path_2m.csv (CCW+ chuẩn), dùng `+omega_odom` thì
        route_state().psi_err/d_m bám ~0 xuyên suốt cua (đúng); dùng
        `-omega_odom` thì lệch tới ~110°/14m (sai, gây _planner_tick_curve
        tái tạo world_yaw sai theo độ cong -> pure_pursuit bùng curvature ->
        xe xoay tại chỗ giữa cua, thẳng thì không lộ vì heading gần như không
        đổi). KHÔNG đổi dấu dòng dưới theo FrenetEKF nữa — 2 file độc lập.
        """
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

    def _update_curve_zone_state(self) -> None:
        """Cập nhật self._in_zone_state (có hysteresis) từ _s_hint mới nhất —
        gọi mỗi khi route_state() chạy (mỗi tick, xem Gps/node.py:_tick).
        correct_gps() chỉ ĐỌC LẠI giá trị này (không tự tính riêng) để cả GPS
        block lẫn in_curve_zone publish ra ngoài dùng chung 1 nguồn trạng thái,
        không lệch nhau. Hysteresis: đã trong zone thì cần vượt qua biên gốc
        thêm curve_zone_hysteresis_m mới coi là ra; chưa trong zone thì vẫn
        dùng đúng biên gốc để vào — tránh nhấp nháy khi s dao động sát biên."""
        if self._s_hint is None or not self.curve_zones:
            self._in_zone_state = False
            return
        s = self._s_hint
        margin = self.config.curve_zone_hysteresis_m
        if self._in_zone_state:
            self._in_zone_state = any(
                a - margin <= s <= b + margin for a, b in self.curve_zones
            )
        else:
            self._in_zone_state = any(a <= s <= b for a, b in self.curve_zones)

    @property
    def in_curve_zone(self) -> bool:
        """Trạng thái curve-zone mới nhất (đã áp hysteresis) — chỉ đúng sau
        khi route_state() đã chạy ít nhất 1 lần tick hiện tại. Gps/node.py
        dùng property này để publish in_curve_zone (phần tử [5] của
        /gps/route_state), thay vì tự tính lại bằng 1 logic riêng có thể
        lệch với cái correct_gps() đang dùng để chặn GPS."""
        return self._in_zone_state

    def correct_gps(
        self,
        lat: float,
        lon: float,
        h_acc_m: float | None = None,
        fix_type: int | None = None,
    ) -> tuple[bool, str]:
        """Correct vị trí bằng fix GPS với R trung thực (= hAcc^2). CHẶN HẲN
        (không correct) khi đang trong curve zone (self._in_zone_state, xem
        _update_curve_zone_state) — cung cua ngắn nên encoder dead-reckon đủ
        chính xác, còn GPS dễ multipath/che khuất đúng lúc rẽ; predict() từ
        encoder một mình quyết định pose suốt zone.

        Trả (accepted, reason). Fix rớt gate không làm hỏng state — bỏ qua thôi.
        """
        if not self.initialized:
            return False, "not-initialized"
        if fix_type is not None and fix_type < self.config.min_fix_type:
            return False, f"fix_type={fix_type} < {self.config.min_fix_type}"
        if h_acc_m is not None and h_acc_m > self.config.max_h_acc_m:
            return False, f"hAcc={h_acc_m:.1f}m > {self.config.max_h_acc_m}m"
        if self._in_zone_state and self.config.curve_zone_block_gps:
            return False, "in-curve-zone (GPS bi chan, chi tin encoder)"

        sigma = self.config.default_h_acc_m if h_acc_m is None else max(0.5, h_acc_m)
        R = np.diag([sigma**2, sigma**2])
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
        """Chốt lệch ngang + heading từ vision — ĐIỀU KIỆN ĐẦU cho đoạn cua.

        Đây là phép GÁN ĐÈ state (không phải Kalman update): x/P bị ghi thẳng,
        không qua gate Mahalanobis nào. Vì vậy CHỈ ĐƯỢC GỌI ĐÚNG 1 LẦN tại
        CẠNH LÊN của curve zone (ngoài -> trong), lúc camera còn thấy line và
        giá trị của nó còn nghĩa. Gọi lặp mỗi tick (hành vi cũ của Gps/node.py)
        làm một mẫu segmentation nhảy làn dịch ngang pose tức thời và reset P
        liên tục -> sigma_d báo "rất chắc chắn" một cách giả tạo. Xem
        Gps/node.py (anchor_on_curve_entry_only) + docstring đầu file đó.

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
        self._update_curve_zone_state()
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
