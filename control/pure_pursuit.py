"""Pure pursuit bám theo /perception/frenet/optimal_path (s,d Frenet).

Thuần Python/numpy, không import rclpy.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PurePursuitConfig:
    lookahead_distance: float = 3    # khớp planner_motion plan_lookahead
    max_linear_speed: float = 1.0
    max_angular_speed: float = 4
    heading_gain: float = 0.5
    track_width: float = 0.3556
    max_wheel_speed: float = 1.2


def _clamp(value: float, limit: float) -> float:
    return max(-limit, min(limit, value))


def compute_cmd_vel(
    path_s: np.ndarray,
    path_d: np.ndarray,
    s_now: float,
    d_now: float,
    psi_now: float,
    target_speed: float,
    cfg: PurePursuitConfig,
    ff_curvature: float = 0.0,
) -> tuple[float, float, float, float]:
    """Trả về (linear_x, angular_z, target_s, target_d) — target_s/target_d là
    điểm lookahead trên path mà pure pursuit đang nhắm tới (để debug/vẽ).
    (0.0, 0.0, s_now, d_now) nếu path không hợp lệ.

    ff_curvature: feed-forward độ cong tuyến [1/m, quy ước panel +=phải],
    cộng thẳng vào curvature TRƯỚC clamp/scale bánh nên vẫn đi qua các giới
    hạn an toàn sẵn có (max_angular_speed, max_wheel_speed). Mặc định 0.0 =
    hành vi y hệt trước đây (reactive thuần); caller (control_node) chỉ
    truyền khác 0 khi đang dùng GPS route trong curve zone — xem Gps/node.py.
    """
    if path_s is None or len(path_s) < 2:
        return 0.0, 0.0, s_now, d_now

    target_s = s_now + cfg.lookahead_distance
    if target_s >= path_s[-1]:
        target_d = float(path_d[-1])
    else:
        target_d = float(np.interp(target_s, path_s, path_d))

    lateral_error = target_d - d_now

    # d_dot = v*sin(psi) (planner_motion/logic.py): psi lớn đã tự kéo lateral
    # error về 0 nhanh -> trừ heading_gain*psi để giảm overshoot/dao động.
    curvature = (
        (2.0 * lateral_error / (cfg.lookahead_distance ** 2))
        - cfg.heading_gain * psi_now
        + ff_curvature
    )

    linear_x = _clamp(target_speed, cfg.max_linear_speed)
    linear_x = max(0.0, linear_x)
    angular_z = _clamp(curvature * linear_x, cfg.max_angular_speed)

    # An toàn: nếu vận tốc bánh suy ra (v ± omega*track_width/2) vượt giới hạn,
    # scale đều cả linear_x và angular_z xuống để giữ đúng tỉ lệ curvature.
    half_track = cfg.track_width / 2.0
    v_left = linear_x - angular_z * half_track
    v_right = linear_x + angular_z * half_track
    peak = max(abs(v_left), abs(v_right))
    if peak > cfg.max_wheel_speed and peak > 0.0:
        scale = cfg.max_wheel_speed / peak
        linear_x *= scale
        angular_z *= scale

    return linear_x, angular_z, target_s, target_d
