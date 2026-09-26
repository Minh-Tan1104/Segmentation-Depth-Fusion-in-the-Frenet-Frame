"""Frenet optimal trajectory planner cho reference THẲNG (khớp perception thật).

Vì line tham chiếu của perception là đường thẳng phía trước (s = đoạn thẳng tiến,
d = lệch ngang so với line, heading ≈ 0), frame toàn cục trùng với frame Frenet:
    x = s (tiến),  y = d (ngang),  yaw ≈ 0.
Nhờ vậy không cần Spline2D / cubic_spline_planner — module này tự chứa, chỉ phụ
thuộc numpy, dùng được trực tiếp trong node ROS.

Quy ước trục ngang (giống panel Frenet trong rgb_dual_model_node):
    +d = bên PHẢI reference,  reference ở d = 0.
    - vị trí ngang của xe:      c_d = -d_meters   (xem _draw_frenet_panel)
    - vị trí ngang obstacle:    d_m               (s = s_m)
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np


@dataclass
class FrenetPlannerConfig:
    # Giới hạn động học (đơn vị SI). Mặc định cỡ robot nhỏ / husky.
    max_speed: float = 2.0          # [m/s]
    max_accel: float = 2.0          # [m/s^2]
    max_curvature: float = 2     # [1/m]
    robot_radius: float = 5     # bán kính an toàn xe + nửa kích thước obstacle [m]

    # Sampling ngang theo bề rộng làn (đối xứng quanh reference)
    max_road_width: float = 3     # nửa bề rộng tối đa lấy mẫu mỗi bên [m]
    d_road_w: float = 0.4          # bước lấy mẫu ngang [m]
    center_offset: float = 0.0    # +d = lệch điểm giữa ưu tiên sang PHẢI [m]

    # Sampling thời gian / vận tốc
    dt: float = 0.2        
    k_obs: float = 10        # bước thời gian [s]
    max_t: float = 4.0              # horizon tối đa [s]
    min_t: float = 3.5              # horizon tối thiểu [s]
    target_speed: float = 2.0       # vận tốc mong muốn [m/s]
    d_t_s: float = 0.5              # bước lấy mẫu vận tốc đích [m/s]
    n_s_sample: int = 1  
    clearance: float = 1.2           # số mẫu vận tốc mỗi phía

  
    k_j: float = 0.1
    k_t: float = 0.1
    k_d: float = 1.0
    k_lat: float = 1.0
    k_lon: float = 1.0

    # Ràng buộc độ dài path tối thiểu
    min_path_length: float = 1.0    # [m]
    min_path_points: int = 3


class QuinticPolynomial:
    def __init__(self, xs, vxs, axs, xe, vxe, axe, t):
        self.a0, self.a1, self.a2 = xs, vxs, axs / 2.0
        A = np.array([[t**3, t**4, t**5],
                      [3 * t**2, 4 * t**3, 5 * t**4],
                      [6 * t, 12 * t**2, 20 * t**3]])
        b = np.array([xe - self.a0 - self.a1 * t - self.a2 * t**2,
                      vxe - self.a1 - 2 * self.a2 * t,
                      axe - 2 * self.a2])
        x = np.linalg.solve(A, b)
        self.a3, self.a4, self.a5 = x[0], x[1], x[2]

    def calc_point(self, t):
        return (self.a0 + self.a1 * t + self.a2 * t**2
                + self.a3 * t**3 + self.a4 * t**4 + self.a5 * t**5)

    def calc_first_derivative(self, t):
        return (self.a1 + 2 * self.a2 * t + 3 * self.a3 * t**2
                + 4 * self.a4 * t**3 + 5 * self.a5 * t**4)

    def calc_second_derivative(self, t):
        return 2 * self.a2 + 6 * self.a3 * t + 12 * self.a4 * t**2 + 20 * self.a5 * t**3

    def calc_third_derivative(self, t):
        return 6 * self.a3 + 24 * self.a4 * t + 60 * self.a5 * t**2


class QuarticPolynomial:
    def __init__(self, xs, vxs, axs, vxe, axe, t):
        self.a0, self.a1, self.a2 = xs, vxs, axs / 2.0
        A = np.array([[3 * t**2, 4 * t**3], [6 * t, 12 * t**2]])
        b = np.array([vxe - self.a1 - 2 * self.a2 * t, axe - 2 * self.a2])
        x = np.linalg.solve(A, b)
        self.a3, self.a4 = x[0], x[1]

    def calc_point(self, t):
        return self.a0 + self.a1 * t + self.a2 * t**2 + self.a3 * t**3 + self.a4 * t**4

    def calc_first_derivative(self, t):
        return self.a1 + 2 * self.a2 * t + 3 * self.a3 * t**2 + 4 * self.a4 * t**3

    def calc_second_derivative(self, t):
        return 2 * self.a2 + 6 * self.a3 * t + 12 * self.a4 * t**2

    def calc_third_derivative(self, t):
        return 6 * self.a3 + 24 * self.a4 * t


class ReferenceCourse:
    """Reference CONG cho planner: spline (x(s), y(s)) qua các waypoint tuyến.

    "generate target course" kiểu frenet_optimal_trajectory.py (bản offline ở
    gốc package) nhưng bằng scipy CubicSpline vì CUBIC_PLANNER không được cài
    qua setup.py. Tham số hoá theo arc-length của polyline đầu vào — truyền
    RouteMapMatcher.route_xy_local (đã smooth) vào đây thì s của course khớp
    trực tiếp với s_m mà RouteEKF publish trên /gps/route_state (cùng polyline,
    cùng cách đo arc-length), nên control_node dùng thẳng s_m làm s0 khi plan.

    Frame local tuyến (mét), yaw CCW+ chuẩn toán học; +d panel (= PHẢI chiều
    đi tuyến) là pháp tuyến (sin yaw, -cos yaw) — khớp RouteEKF.initialize().
    Sample sẵn lưới mịn ds để position/yaw/kappa tra bằng nội suy tuyến tính.
    """

    def __init__(self, xy: np.ndarray, ds: float = 0.5) -> None:
        from scipy.interpolate import CubicSpline

        pts = np.asarray(xy, dtype=float)
        if len(pts) < 2:
            raise ValueError("ReferenceCourse can it nhat 2 diem")
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        if np.any(seg <= 0.0):
            raise ValueError("ReferenceCourse co 2 diem lien tiep trung nhau")
        s_knots = np.concatenate([[0.0], np.cumsum(seg)])
        csx = CubicSpline(s_knots, pts[:, 0])
        csy = CubicSpline(s_knots, pts[:, 1])
        self.total_length_m = float(s_knots[-1])

        n = max(2, int(math.ceil(self.total_length_m / max(ds, 1e-3))) + 1)
        self._s = np.linspace(0.0, self.total_length_m, n)
        self._x = csx(self._s)
        self._y = csy(self._s)
        dx, dy = csx(self._s, 1), csy(self._s, 1)
        ddx, ddy = csx(self._s, 2), csy(self._s, 2)
        # unwrap để nội suy heading không gãy khi atan2 nhảy ±pi giữa 2 mẫu
        # (chỉ dùng qua sin/cos nên không cần wrap lại khi tra).
        self._yaw = np.unwrap(np.arctan2(dy, dx))
        denom = np.power(dx * dx + dy * dy, 1.5)
        self._kappa = (dx * ddy - dy * ddx) / np.where(denom < 1e-9, 1e-9, denom)

    def position(self, s):
        """(x, y) local tại s (scalar hoặc array, clip trong [0, L])."""
        s = np.clip(s, 0.0, self.total_length_m)
        return np.interp(s, self._s, self._x), np.interp(s, self._s, self._y)

    def yaw(self, s):
        """Góc tiếp tuyến [rad, CCW+] tại s (scalar hoặc array)."""
        s = np.clip(s, 0.0, self.total_length_m)
        return np.interp(s, self._s, self._yaw)

    def kappa_ccw(self, s_m: float) -> float:
        """Độ cong CÓ DẤU tại s [1/m, CCW+ = cua trái] — đổi dấu ở chỗ dùng
        nếu cần quy ước panel (+ = phải), giống Gps/node.py:_curve_ff_kappa."""
        s = np.clip(s_m, 0.0, self.total_length_m)
        return float(np.interp(s, self._s, self._kappa))

    def project(self, x: float, y: float) -> tuple[float, float, float]:
        """Chiếu điểm map-local (x, y) lên course -> (s, d, yaw_ref).

        BẮT BUỘC dùng khi state (s, d, psi_err) từ RouteEKF sang curve mode:
        RouteEKF đo d/psi_err so với POLYLINE (RouteMapMatcher, tiếp tuyến đoạn
        bậc thang ~2m), còn planner + reconstruct pose ở đây dùng SPLINE — 2
        tham chiếu lệch tới ~7° heading ngay chỗ cua gấp (đo bằng offline
        script). Chiếu lại pose lên chính course này để (s, d, psi_err) NHẤT
        QUÁN với reference mà planner/pure pursuit dùng, thay vì trộn 2 frame.

        +d = PHẢI chiều đi course (panel), khớp _calc_global_paths_curved /
        RouteEKF.initialize(). Nearest-sample rồi refine trên 2 đoạn kề để đạt
        độ chính xác dưới bước sample (ds)."""
        d2 = (self._x - x) ** 2 + (self._y - y) ** 2
        i = int(np.argmin(d2))
        best_s = float(self._s[i])
        best_dist2 = float(d2[i])
        for j in (i - 1, i):
            if j < 0 or j + 1 >= len(self._s):
                continue
            ax, ay = self._x[j], self._y[j]
            vx, vy = self._x[j + 1] - ax, self._y[j + 1] - ay
            L2 = vx * vx + vy * vy
            if L2 <= 1e-12:
                continue
            t = ((x - ax) * vx + (y - ay) * vy) / L2
            t = min(1.0, max(0.0, t))
            cx, cy = ax + t * vx, ay + t * vy
            dist2 = (x - cx) ** 2 + (y - cy) ** 2
            if dist2 <= best_dist2:
                best_dist2 = dist2
                best_s = float(self._s[j] + t * (self._s[j + 1] - self._s[j]))
        # d theo pháp tuyến của SPLINE tại best_s (nhất quán yaw/position dùng
        # ở _calc_global_paths_curved), không phải pháp tuyến đoạn thô.
        th = float(self.yaw(best_s))
        rx, ry = self.position(best_s)
        d = (x - float(rx)) * math.sin(th) - (y - float(ry)) * math.cos(th)
        return best_s, float(d), th


@dataclass
class FrenetPath:
    t: np.ndarray = field(default_factory=lambda: np.array([]))
    d: np.ndarray = field(default_factory=lambda: np.array([]))
    d_d: np.ndarray = field(default_factory=lambda: np.array([]))
    d_dd: np.ndarray = field(default_factory=lambda: np.array([]))
    d_ddd: np.ndarray = field(default_factory=lambda: np.array([]))
    s: np.ndarray = field(default_factory=lambda: np.array([]))
    s_d: np.ndarray = field(default_factory=lambda: np.array([]))
    s_dd: np.ndarray = field(default_factory=lambda: np.array([]))
    s_ddd: np.ndarray = field(default_factory=lambda: np.array([]))
    cd: float = 0.0
    cv: float = 0.0
    cf: float = 0.0
    # Cartesian (straight reference): x = s, y = d
    x: np.ndarray = field(default_factory=lambda: np.array([]))
    y: np.ndarray = field(default_factory=lambda: np.array([]))
    yaw: np.ndarray = field(default_factory=lambda: np.array([]))
    c: np.ndarray = field(default_factory=lambda: np.array([]))


class FrenetOptimalPlanner:
    """Lập kế hoạch Frenet trên reference thẳng. Gọi `plan(...)` mỗi frame."""

    def __init__(self, config: FrenetPlannerConfig | None = None):
        self.config = config or FrenetPlannerConfig()

    # ── sinh các quỹ đạo ứng viên ───────────────────────────────────────────
    def _calc_frenet_paths(self, c_speed, c_d, c_d_d, c_d_dd, s0):
        cfg = self.config
        paths = []
        # Đối xứng quanh center_offset, dựng bằng SỐ BƯỚC nguyên (không phải
        # cộng biên rồi arange) -> luôn có đúng 1 candidate di == center_offset
        # bất kể d_road_w/max_road_width là bao nhiêu. Trước đây hardcode
        # [-3+center_offset, 3+center_offset) hoặc [-2+center_offset, ...) —
        # khi biên KHÔNG phải bội số của d_road_w tính từ center_offset (vd
        # -3 với d_road_w=0.4), giá trị đúng-giữa-làn bị "lọt lưới" hoàn
        # toàn, planner không bao giờ chọn được path về đúng tâm (chỉ có
        # ±d_road_w/2 gần nhất) — đã xảy ra thật, xem lịch sử debug. Dùng
        # max_road_width (plan_road_width) làm nửa bề rộng thay vì hardcode.
        n_steps = max(1, int(round(cfg.max_road_width / cfg.d_road_w)))
        di_values = cfg.center_offset + cfg.d_road_w * np.arange(-n_steps, n_steps + 1)
        # Gồm cả max_t (+dt/2 chống sai số float): arange(min_t, max_t) bỏ
        # mất max_t, trong khi RL chọn Ti liên tục trong [min_t, max_t].
        Ti_values = np.arange(cfg.min_t, cfg.max_t + cfg.dt / 2, cfg.dt)
        tv_values = np.arange(
            cfg.target_speed - cfg.d_t_s * cfg.n_s_sample,
            cfg.target_speed + cfg.d_t_s * cfg.n_s_sample + 1e-9,
            cfg.d_t_s,
        )

        for di in di_values:
            for Ti in Ti_values:
                lat_qp = QuinticPolynomial(c_d, c_d_d, c_d_dd, di, 0.0, 0.0, Ti)
                t = np.arange(0.0, Ti, cfg.dt)
                d = lat_qp.calc_point(t)
                d_d = lat_qp.calc_first_derivative(t)
                d_dd = lat_qp.calc_second_derivative(t)
                d_ddd = lat_qp.calc_third_derivative(t)

                for tv in tv_values:
                    lon_qp = QuarticPolynomial(s0, c_speed, 0.0, tv, 0.0, Ti)
                    s = lon_qp.calc_point(t)
                    s_d = lon_qp.calc_first_derivative(t)
                    s_dd = lon_qp.calc_second_derivative(t)
                    s_ddd = lon_qp.calc_third_derivative(t)

                    fp = FrenetPath(
                        t=t, d=d, d_d=d_d, d_dd=d_dd, d_ddd=d_ddd,
                        s=s, s_d=s_d, s_dd=s_dd, s_ddd=s_ddd,
                    )
                    fp.cd, fp.cv, fp.cf = FrenetOptimalPlanner._path_cost(fp, cfg, Ti)
                    paths.append(fp)
        return paths

    # ── 1 quỹ đạo DUY NHẤT cho (di, Ti, tv) đã chọn sẵn — đúng thân vòng lặp
    # _calc_frenet_paths nhưng không enumerate. Dùng khi policy RL (plan_use_rl,
    # planner_motion/rl_policy.py) đã chọn thẳng (d_target, Ti) thay cho argmin
    # cost. Trả path frame Frenet (chưa có x/y — caller tự _calc_global_paths).
    def build_path(self, s0, c_speed, c_d, c_d_d, c_d_dd, di, Ti, tv):
        cfg = self.config
        lat_qp = QuinticPolynomial(c_d, c_d_d, c_d_dd, di, 0.0, 0.0, Ti)
        lon_qp = QuarticPolynomial(s0, c_speed, 0.0, tv, 0.0, Ti)
        t = np.arange(0.0, Ti, cfg.dt)
        fp = FrenetPath(
            t=t,
            d=lat_qp.calc_point(t),
            d_d=lat_qp.calc_first_derivative(t),
            d_dd=lat_qp.calc_second_derivative(t),
            d_ddd=lat_qp.calc_third_derivative(t),
            s=lon_qp.calc_point(t),
            s_d=lon_qp.calc_first_derivative(t),
            s_dd=lon_qp.calc_second_derivative(t),
            s_ddd=lon_qp.calc_third_derivative(t),
        )
        fp.cd, fp.cv, fp.cf = FrenetOptimalPlanner._path_cost(fp, cfg, Ti)
        return fp

    # ── cost (cd, cv, cf) của 1 path đơn — tách khỏi vòng lặp enumerate ở
    # trên để nơi khác (vd train_frenet_rl.py: RL thay phần CHỌN di nhưng
    # vẫn cần đánh giá cost ĐÚNG công thức gốc cho path mà nó tự sinh) gọi
    # lại được mà không viết lại công thức. Hành vi enumerate phía trên
    # không đổi — chỉ tách phép tính, số ra giống hệt trước.
    @staticmethod
    def _path_cost(fp: "FrenetPath", cfg: "FrenetPlannerConfig", Ti: float) -> tuple[float, float, float]:
        Jp = np.sum(fp.d_ddd**2)
        Js = np.sum(fp.s_ddd**2)
        ds = (cfg.target_speed - fp.s_d[-1]) ** 2

        cd = cfg.k_j * Jp + cfg.k_t * Ti + cfg.k_d * (fp.d[-1] - cfg.center_offset) ** 2
        cv = cfg.k_j * Js + cfg.k_t * Ti + cfg.k_d * ds
        cf = cfg.k_lat * cd + cfg.k_lon * cv
        return cd, cv, cf

    # ── yaw/độ cong numeric từ (fp.x, fp.y) — dùng chung thẳng lẫn cong ──────
    @staticmethod
    def _finalize_cartesian(fp):
        if len(fp.x) < 2:
            return
        dx = np.diff(fp.x)
        dy = np.diff(fp.y)
        fp.yaw = np.append(np.arctan2(dy, dx), 0.0)
        fp.yaw[-1] = fp.yaw[-2]
        ds = np.hypot(dx, dy)
        ds = np.append(ds, ds[-1])
        # wrap dyaw về (-pi, pi]: heading map-frame có thể nhảy qua ±pi giữa
        # 2 điểm (reference cong) — không wrap thì fp.c vọt ~2pi/ds giả tạo,
        # path nào đi qua chỗ đó bị loại oan ở max_curvature check. Với
        # reference thẳng dyaw luôn nhỏ, wrap là no-op, hành vi y cũ.
        dyaw = np.diff(fp.yaw)
        dyaw = (dyaw + np.pi) % (2.0 * np.pi) - np.pi
        fp.c = dyaw / np.where(ds[:-1] == 0, 1e-9, ds[:-1])

    # ── chuyển sang Cartesian (reference thẳng: x=s, y=d) ────────────────────
    @staticmethod
    def _calc_global_paths(paths):
        for fp in paths:
            fp.x = fp.s.copy()
            fp.y = fp.d.copy()
            FrenetOptimalPlanner._finalize_cartesian(fp)
        return paths

    # ── chuyển sang Cartesian theo reference CONG (frame local tuyến CSV) ────
    @staticmethod
    def _calc_global_paths_curved(paths, course: "ReferenceCourse"):
        for fp in paths:
            rx, ry = course.position(fp.s)
            ryaw = course.yaw(fp.s)
            # +d panel = PHẢI chiều đi tuyến -> pháp tuyến (sin yaw, -cos yaw),
            # khớp RouteEKF.initialize()/anchor_lateral() (Gps/route_ekf.py).
            fp.x = rx + fp.d * np.sin(ryaw)
            fp.y = ry - fp.d * np.cos(ryaw)
            FrenetOptimalPlanner._finalize_cartesian(fp)
        return paths

    # ── kiểm tra va chạm (obstacle dạng điểm + bán kính) ─────────────────────
    def _collision_free(self, fp, obstacles):
        if obstacles is None or len(obstacles) == 0:
            return True
        r2 = self.config.robot_radius ** 2
        for ox, oy in obstacles:
            d2 = (fp.x - ox) ** 2 + (fp.y - oy) ** 2
            if np.any(d2 <= r2):
                return False
        return True
    def _obstacle_cost(self, fp, obstacles):
        if obstacles is None or len(obstacles) == 0:
            return 0.0
        min_d2 = np.inf
        for ox, oy in obstacles:
            d2 = np.min((fp.x - ox) ** 2 + (fp.y - oy) ** 2)
            if d2 < min_d2:
                min_d2 = d2
        gap = self.config.clearance - math.sqrt(min_d2)
        return gap if gap > 0.0 else 0.0   # chỉ phạt khi gần hơn clearance
    def _check_paths(self, paths, obstacles, ref_kappa=None):
        cfg = self.config
        ok = []
        if ref_kappa is not None:
            s_grid, k_grid = ref_kappa
        for fp in paths:
            if len(fp.x) < cfg.min_path_points:
                continue
            if np.any(fp.s_d > cfg.max_speed):
                continue
            if np.any(np.abs(fp.s_dd) > cfg.max_accel):
                continue
            if len(fp.c):
                # fp.c là độ cong CỦA PATH trong frame (x=s, y=d phải+) — cùng
                # quy ước dấu panel với kappa từ Gps/node.py (+ = cua phải)
                # nên cộng thẳng được: path bám sát reference cong (kappa
                # lớn) cần path curvature bù lại gần bằng -ref_kappa mới coi
                # là "thẳng" so với đường thật, ref_kappa=None (mặc định) giữ
                # hành vi cũ (so max_curvature với thẳng fp.c thô).
                c_total = fp.c
                if ref_kappa is not None:
                    c_total = fp.c + np.interp(fp.s[: len(fp.c)], s_grid, k_grid)
                if np.any(np.abs(c_total) > cfg.max_curvature):
                    continue
            if not self._collision_free(fp, obstacles):
                continue
            ok.append(fp)
        return ok

    def plan(self, s0, c_speed, c_d, c_d_d, c_d_dd, obstacles, ref_kappa=None, course=None):
        """Trả về (best_path, candidate_paths). best_path=None nếu không khả thi.

        obstacles: iterable các (x=s, y=d) [m] trong frame Frenet thẳng — hoặc
        (x, y) map-local khi có course (xem dưới).
        ref_kappa: (s_grid, k_grid) độ cong CSV dọc quãng đường phía trước
        (frame tương đối, s_grid[0]=0 tại vị trí hiện tại — xem Gps/node.py:
        _curve_kappa_profile qua control/node.py), dùng để đánh giá
        max_curvature so với hình dạng cua THẬT thay vì đường thẳng. None
        (mặc định) = hành vi y cũ.
        course: ReferenceCourse (reference CONG từ spline tuyến CSV) — khi khác
        None: s0 là s TUYỆT ĐỐI trên tuyến, c_d là lệch ngang SO VỚI TUYẾN
        (panel, +=phải), obstacles là (x, y) map-local, fp.x/fp.y trả ra là
        toạ độ map-local, và max_curvature check trên độ cong path THẬT trong
        map frame (đã bao gồm độ cong reference) — ref_kappa bị bỏ qua vì
        không còn cần hack bù nữa.
        """
        paths = self._calc_frenet_paths(c_speed, c_d, c_d_d, c_d_dd, s0)
        if course is not None:
            paths = self._calc_global_paths_curved(paths, course)
            paths = self._check_paths(paths, obstacles, None)
        else:
            paths = self._calc_global_paths(paths)
            paths = self._check_paths(paths, obstacles, ref_kappa)

        best, mincost = None, float("inf")
        for fp in paths:
            cost = fp.cf + self.config.k_obs * self._obstacle_cost(fp, obstacles)
            if cost <= mincost:
                mincost = cost
                best = fp
        return best, paths
