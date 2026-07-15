from __future__ import annotations

import math
from typing import Any

import numpy as np

from .frenet_planner import FrenetOptimalPlanner, FrenetPlannerConfig


class PlannerLogic:
    """Frenet Optimal Planner (reference thẳng) — thuần Python/numpy, không rclpy.

    Sinh hàng trăm quỹ đạo ứng viên mỗi lần gọi `plan()` (polynomial bậc 5) —
    tốn CPU đáng kể, đây là lý do tách thành node riêng (`planner_motion/node.py`)
    khỏi `perception_node`.
    """

    def __init__(
        self,
        plan_enable: bool,
        plan_speed: float,
        plan_robot_radius: float,
        plan_road_width: float,
        plan_max_curvature: float,
        plan_lookahead: float,
        plan_clearance: float = 1.2,
        plan_obstacle_weight: float = 10.0,
        plan_center_offset: float = 0.0,
        plan_center_weight: float = 1.0,
        plan_min_horizon_s: float = 3.5,
        plan_max_horizon_s: float = 4.0,
        plan_d_road_w: float = 0.4,
    ) -> None:
        self.plan_enable = plan_enable
        self.plan_lookahead = plan_lookahead
        self.planner = FrenetOptimalPlanner(
            FrenetPlannerConfig(
                target_speed=plan_speed,
                max_speed=plan_speed * 1.5,
                robot_radius=plan_robot_radius,
                max_road_width=plan_road_width,
                max_curvature=plan_max_curvature,
                # robot_radius = ràng buộc CỨNG (path đi gần hơn bị loại hẳn).
                # clearance/k_obs = phạt MỀM — path vẫn được phép đi gần hơn
                # clearance nếu không còn lựa chọn khác, nhưng cost tăng dần
                # khi gần obstacle hơn, k_obs lớn hơn = ưu tiên né xa mạnh hơn
                # các yếu tố khác (jerk, lệch reference, thời gian).
                clearance=plan_clearance,
                k_obs=plan_obstacle_weight,
                center_offset=plan_center_offset,
                # k_d nhân trực tiếp (d[-1]-center_offset)^2 trong cd — tăng
                # lên = phạt nặng hơn việc kết thúc path lệch center_offset,
                # planner chấp nhận jerk ngang lớn hơn để về gần tâm hơn.
                # ĐÃ THỬ k_lat trước (nhân đều cả cụm cd vs cv) nhưng không ăn
                # thua: trade-off "về tâm vs jerk" nằm BÊN TRONG cd (do k_d),
                # nhân đều cả cụm không đổi được thứ tự ưu tiên đó. Lưu ý k_d
                # cũng nhân "ds" (lệch target_speed) trong cv — tăng k_d cũng
                # làm planner bám tốc độ mục tiêu chặt hơn, không chỉ mỗi vị
                # trí. Xem frenet_planner.py: cd = k_j*Jp+k_t*Ti+k_d*(d-off)^2,
                # cv = k_j*Js+k_t*Ti+k_d*ds.
                k_d=plan_center_weight,
                # Horizon thời gian cho polynomial ngang (di_values sinh trong
                # khoảng [min_t, max_t)) — khi lệch ngang lớn + plan_center_weight
                # cao ép path về hẳn center_offset, quay về TRONG THỜI GIAN NGẮN
                # đòi hỏi độ cong lớn ở đoạn đầu path. Kéo dài horizon này cho
                # cùng quãng đường ngang cần đi thì độ cong giảm hẳn (đã đo: c_d=1.5m,
                # k_d=4.0 -> max|c| 0.237 ở max_t=4.0s, còn 0.122 ở max_t=6.0s,
                # vẫn về hẳn center_offset) — đánh đổi: s_max path dài hơn tương ứng.
                min_t=plan_min_horizon_s,
                max_t=plan_max_horizon_s,
                # Bước nhảy khi lấy mẫu di (lệch ngang ứng viên) — di_values =
                # arange(-2+center_offset, 3+center_offset, d_road_w). Giảm
                # xuống = nhiều ứng viên hơn, chọn di mịn hơn (đỡ "nhảy bậc"
                # như đã thấy với plan_center_weight: 0.0 hoặc 0.4, không có
                # mức giữa) nhưng tốn CPU hơn (số path ~ tỉ lệ nghịch d_road_w).
                # LƯU Ý: phạm vi [-2, 3] hiện hardcode, KHÔNG dùng plan_road_width.
                d_road_w=plan_d_road_w,
            )
        )

    @property
    def target_speed(self) -> float:
        return self.planner.config.target_speed

    def plan(self, frenet: dict[str, Any], detections: list[dict[str, Any]]):
        """Chạy Frenet optimal planner (reference thẳng) từ frenet đo được trực
        tiếp (perception). Trục ngang theo panel: +d = phải reference. Xe ở
        c_d = -d_meters. Gắn kết quả vào frenet["optimal_path"] để vẽ.
        Trả về best_path (hoặc None) để caller publish thêm Float64MultiArray.

        Dùng cho viz/rosbag (đường cũ qua frenet_state topic) — pipeline live
        dùng `plan_from_state()` qua EKF, xem control/node.py.
        """
        d_m = float(frenet.get("d_meters_filtered", frenet.get("d_meters", 0.0)))
        hdg = math.radians(float(frenet.get("heading_filtered", frenet.get("heading_deg", 0.0))))
        c_d = -d_m
        c_d_d = self.target_speed * math.sin(hdg)   # vận tốc ngang theo heading (panel frame)
        best, extra = self.plan_from_state(c_d, c_d_d, detections)
        frenet.update(extra)
        return best

    def plan_from_state(
        self, c_d: float, c_d_d: float, detections: list[dict[str, Any]]
    ) -> tuple[Any, dict[str, Any]]:
        """Chạy planner từ pose (c_d, c_d_d) cho trước, không cần frenet đo
        trực tiếp từ camera — dùng được cả khi mất line (c_d/c_d_d lấy từ EKF
        dead-reckon). Trả về (best_path, frenet_extra) — frenet_extra là dict
        optimal_path/optimal_target_d/candidate_paths để publish/vẽ.
        """
        frenet_extra: dict[str, Any] = {
            "optimal_path": None,
            "optimal_target_d": None,
            "candidate_paths": [],
        }
        if not self.plan_enable:
            return None, frenet_extra

        c_speed = self.target_speed

        # Obstacle: x_m/s_m đo trực tiếp theo frame XE (camera, không cần line
        # — xem perception/logic.py: _annotate_detection_frenet luôn tính
        # được kể cả khi mất line). Quy về panel frame (so với lane reference)
        # bằng cách cộng c_d (pose xe so với lane hiện tại, từ EKF hoặc đo
        # trực tiếp) làm trung gian — KHÔNG còn phụ thuộc spline của line.
        obstacles = []
        for det in detections:
            fr = det.get("frenet")
            if not fr or not fr.get("available"):
                continue
            s_o = float(fr.get("s_m_filtered", fr.get("s_m")))
            x_o = float(fr.get("x_m_filtered", fr.get("x_m")))
            d_o = x_o + c_d
            obstacles.append((s_o, d_o))

        best, candidates = self.planner.plan(0.0, c_speed, c_d, c_d_d, 0.0, obstacles)

        # Many candidates share the same lateral target (di) but differ only in
        # duration/speed, so they all converge onto the same endpoint. Keep just
        # the lowest-cost path per distinct di to avoid that bunched-up look.
        by_target_d: dict[float, Any] = {}
        for fp in candidates:
            if fp is best or len(fp.d) == 0:
                continue
            key = round(float(fp.d[-1]), 1)
            if key not in by_target_d or fp.cf < by_target_d[key].cf:
                by_target_d[key] = fp
        frenet_extra["candidate_paths"] = [
            list(zip(fp.d.tolist(), fp.s.tolist())) for fp in by_target_d.values()
        ]
        if best is None:
            return None, frenet_extra

        # path dạng (d, s) cho panel + target_d (theo convention d_meters: phải dương)
        frenet_extra["optimal_path"] = list(zip(best.d.tolist(), best.s.tolist()))
        idx = int(np.argmin(np.abs(best.s - self.plan_lookahead))) if len(best.s) else 0
        target_d_panel = float(best.d[idx]) if len(best.d) else 0.0
        frenet_extra["optimal_target_d"] = -target_d_panel  # đổi về convention d_meters
        return best, frenet_extra

    def best_path_to_flat_array(self, best_path) -> list[float]:
        """Flatten [s0, d0, s1, d1, ...] trong frame Frenet thẳng (d: +phải reference)."""
        if best_path is None:
            return []
        data: list[float] = []
        for s_v, d_v in zip(best_path.s.tolist(), best_path.d.tolist()):
            data.extend([float(s_v), float(d_v)])
        return data

    def best_path_target_d(self, best_path) -> float | None:
        """target_d theo convention d_meters (phải dương), hoặc None nếu không có path."""
        if best_path is None:
            return None
        idx = int(np.argmin(np.abs(best_path.s - self.plan_lookahead))) if len(best_path.s) else 0
        return -float(best_path.d[idx])
