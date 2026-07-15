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
        Ti_values = np.arange(cfg.min_t, cfg.max_t, cfg.dt)
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

                    Jp = np.sum(d_ddd**2)
                    Js = np.sum(s_ddd**2)
                    ds = (cfg.target_speed - s_d[-1]) ** 2

                    cd = cfg.k_j * Jp + cfg.k_t * Ti + cfg.k_d * (d[-1] - cfg.center_offset) ** 2
                    cv = cfg.k_j * Js + cfg.k_t * Ti + cfg.k_d * ds

                    fp = FrenetPath(
                        t=t, d=d, d_d=d_d, d_dd=d_dd, d_ddd=d_ddd,
                        s=s, s_d=s_d, s_dd=s_dd, s_ddd=s_ddd,
                        cd=cd, cv=cv, cf=cfg.k_lat * cd + cfg.k_lon * cv,
                    )
                    paths.append(fp)
        return paths

    # ── chuyển sang Cartesian (reference thẳng: x=s, y=d) ────────────────────
    @staticmethod
    def _calc_global_paths(paths):
        for fp in paths:
            fp.x = fp.s.copy()
            fp.y = fp.d.copy()
            if len(fp.x) < 2:
                continue
            dx = np.diff(fp.x)
            dy = np.diff(fp.y)
            fp.yaw = np.append(np.arctan2(dy, dx), 0.0)
            fp.yaw[-1] = fp.yaw[-2]
            ds = np.hypot(dx, dy)
            ds = np.append(ds, ds[-1])
            dyaw = np.diff(fp.yaw)
            fp.c = dyaw / np.where(ds[:-1] == 0, 1e-9, ds[:-1])
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
    def _check_paths(self, paths, obstacles):
        cfg = self.config
        ok = []
        for fp in paths:
            if len(fp.x) < cfg.min_path_points:
                continue
            if np.any(fp.s_d > cfg.max_speed):
                continue
            if np.any(np.abs(fp.s_dd) > cfg.max_accel):
                continue
            if len(fp.c) and np.any(np.abs(fp.c) > cfg.max_curvature):
                continue
            if not self._collision_free(fp, obstacles):
                continue
            ok.append(fp)
        return ok

    def plan(self, s0, c_speed, c_d, c_d_d, c_d_dd, obstacles):
        """Trả về (best_path, candidate_paths). best_path=None nếu không khả thi.

        obstacles: iterable các (x=s, y=d) [m] trong frame Frenet thẳng.
        """
        paths = self._calc_frenet_paths(c_speed, c_d, c_d_d, c_d_dd, s0)
        paths = self._calc_global_paths(paths)
        paths = self._check_paths(paths, obstacles)

        best, mincost = None, float("inf")
        for fp in paths:
            cost = fp.cf + self.config.k_obs * self._obstacle_cost(fp, obstacles)
            if cost <= mincost:
                mincost = cost
                best = fp
        return best, paths
