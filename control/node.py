"""ROS2 node: chuyển mode manual/auto (nút R1) và publish /cmd_vel.

- manual: cần gạt tay cầm PS4 (axes[1]=ga, axes[3]=lái) -> /cmd_vel trực tiếp.
- auto: EKF fusion (encoder /odom + perception /perception/frenet_state)
  + pure pursuit bám path tự tính nội bộ (xem _planner_tick) -> /cmd_vel.

control_node tự sở hữu Frenet Optimal Planner (PlannerLogic) thay vì chỉ tiêu
thụ /perception/frenet/optimal_path — lý do: planner_motion_node chỉ replan
khi có frenet đo trực tiếp từ camera (mất line = path rỗng = không có gì để
bám). Ở đây EKF predict() mỗi tick từ /odom độc lập với perception, nên
planner nội bộ có thể replan bằng (c_d, c_d_d) lấy từ EKF dead-reckon ngay cả
khi mất line — pure pursuit luôn có path để bám, không phụ thuộc tần suất
camera thấy line. planner_motion_node vẫn chạy song song như cũ, chỉ phục vụ
visualization/rosbag (xem planner_motion/logic.py: plan() vs plan_from_state()).

Toggle mode bằng cạnh lên của nút R1 (mặc định buttons[5], qua joy_pygame_node).
Node này KHÔNG đụng tới serial — chỉ publish /cmd_vel, encoder_node mới là nơi
sở hữu cổng serial và chuyển /cmd_vel thành lệnh hoverboard.

GPS assist (tắt mặc định, gps_assist_enable): kích hoạt khi GPS đủ tin cậy
(sigma_d < gps_max_sigma_d) VÀ (mất line quá gps_vision_stale_s HOẶC — nếu
bật gps_force_in_curve_zone — đang ở trong 1 curve zone đã ghi nhận trên bản
đồ, kể cả khi camera vẫn còn thấy line). Mặc định gps_force_in_curve_zone=
False thì hành vi y hệt trước: chỉ chuyển khi thật sự mất line. Bật True thì
CHỦ ĐỘNG chuyển sang GPS+encoder ngay khi vào cua đã biết trước trên bản đồ,
không cần chờ mất line — xem in_curve_zone (phần tử thứ 6 của
/gps/route_state, Gps/node.py). Khi kích hoạt, _planner_tick trộn d từ GPS
vào d của FrenetEKF theo gps_d_gain (mặc định 1.0 = tin GPS hoàn toàn, 0.0 =
bỏ hẳn GPS d chỉ còn FrenetEKF, ở giữa = trộn tuyến tính) — vẫn cùng 1 pure
pursuit, chỉ đổi nguồn d (và c_d_d suy ra từ đó). psi (heading) LUÔN lấy từ
FrenetEKF (encoder + camera correct), GPS KHÔNG BAO GIỜ ghi đè heading —
tránh 2 nguồn heading độc lập (FrenetEKF vs RouteEKF) trôi lệch nhau. Còn
line thì GPS không bao giờ được đụng vào lệch ngang (trừ khi force+in_zone).
GPS node hoàn toàn không publish /cmd_vel, không có quyền lái xe trực tiếp.

Curvature feed-forward (gps_ff_gain, mặc định 1.0): phần tử thứ 5 của
/gps/route_state (kappa_ff, xem Gps/node.py) là độ cong CSV feed-forward,
CHỈ khác 0 trong 1 curve zone của tuyến. compute_cmd_vel (pure_pursuit.py)
cộng gps_ff_gain*kappa_ff vào curvature TRƯỚC clamp, và CHỈ khi
_use_gps_route() true — cùng gate với (d, psi) ở trên, không phải điều kiện
riêng. gps_ff_gain=0.0 tắt hẳn phần feed-forward mà không tắt cả GPS assist.

Curve-Frenet mode (curve_frenet_enable): trong curve zone, thay toàn bộ cụm
"d-blend + kappa_ff hack" ở trên bằng kiến trúc reference CONG đúng nghĩa:
  - State (s, d, psi_err) lấy TRỰC TIẾP từ RouteEKF (/gps/route_state) — s
    tuyệt đối trên tuyến CSV (mốc từ map-match, encoder tiến), d được
    anchor_lateral() từ vision NGAY TRƯỚC lúc vào zone (Gps/route_ekf.py),
    psi_err so với tiếp tuyến tuyến nên giữ nhỏ suốt cua (không phình như
    psi FrenetEKF frame thẳng).
  - Planner chạy trên ReferenceCourse (spline CSV, planner_motion/
    frenet_planner.py) — candidate path bám đúng hình học cua, obstacle
    transform sang map-local, max_curvature check trên path thật.
  - Pure pursuit dùng cùng state + feed-forward = độ cong course tại đúng
    điểm lookahead.
  - Ra khỏi zone (in_curve_zone tắt, đã hysteresis bên RouteEKF): bàn giao
    lại vision — FrenetEKF.reset_lateral(d, psi_err cuối từ route) rồi
    vision correct tiếp như cũ. Xem _planner_tick/_planner_tick_curve.
"""
from __future__ import annotations

import csv
import enum
import json
import math
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Joy
from std_msgs.msg import Float64MultiArray, String

from planner_motion.logic import PlannerLogic

from .ekf import FrenetEKF
from .pure_pursuit import PurePursuitConfig, compute_cmd_vel


def _wrap(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


class Mode(enum.Enum):
    MANUAL = "manual"
    AUTO = "auto"


def _apply_deadzone(value: float, deadzone: float) -> float:
    """Loại nhiễu quanh tâm cần analog, rescale lại về full dải -1..1 ngay
    sau vùng chết (giống Encoder/wireless.py) — tránh xe tự trôi khi không
    chạm cần."""
    if abs(value) < deadzone:
        return 0.0
    sign = 1.0 if value > 0 else -1.0
    return sign * (abs(value) - deadzone) / (1.0 - deadzone)


class ControlNode(Node):
    def __init__(self) -> None:
        super().__init__("control_node")

        self.declare_parameter("control_rate_hz", 50.0)
        self.declare_parameter("joy_topic", "/joy")
        self.declare_parameter("r1_button_index", 5)
        self.declare_parameter("max_linear_speed", 1.0)
        self.declare_parameter("max_angular_speed", 1.5)
        self.declare_parameter("manual_speed_axis", 1)
        self.declare_parameter("manual_steer_axis", 5)
        self.declare_parameter("manual_invert_speed_axis", True)
        self.declare_parameter("manual_invert_steer_axis", False)
        self.declare_parameter("manual_deadzone", 0.15)
        self.declare_parameter("auto_speed", 2.0)
        self.declare_parameter("lookahead_distance", 2.0)
        self.declare_parameter("pure_pursuit_heading_gain", 0.5)
        self.declare_parameter("track_width", 0.3556)
        self.declare_parameter("max_wheel_speed", 1.2)
        self.declare_parameter("ekf_q_d", 0.01)
        self.declare_parameter("ekf_q_psi", 0.01)
        self.declare_parameter("ekf_r_d", 0.04)
        self.declare_parameter("ekf_r_psi", math.radians(5.0) ** 2)
        self.declare_parameter("planner_rate_hz", 15.0)
        self.declare_parameter("plan_enable", True)
        self.declare_parameter("plan_speed", 2.0)
        self.declare_parameter("plan_robot_radius", 0.6)
        self.declare_parameter("plan_road_width", 2.5)
        self.declare_parameter("plan_max_curvature", 1.5)
        self.declare_parameter("plan_lookahead", 1.5)
        self.declare_parameter("plan_clearance", 1.2)
        self.declare_parameter("plan_obstacle_weight", 10.0)
        self.declare_parameter("plan_center_offset", 0.0)
        self.declare_parameter("plan_center_weight", 1.0)
        self.declare_parameter("plan_min_horizon_s", 3.5)
        self.declare_parameter("plan_max_horizon_s", 4.0)
        self.declare_parameter("plan_d_road_w", 0.4)
        self.declare_parameter("gps_assist_enable", False)
        self.declare_parameter("gps_vision_stale_s", 1.0)
        self.declare_parameter("gps_max_sigma_d", 0.6)
        self.declare_parameter("gps_ff_gain", 1.0)
        self.declare_parameter("gps_d_gain", 1.0)
        self.declare_parameter("gps_force_in_curve_zone", False)
        self.declare_parameter("curve_frenet_enable", False)
        # Lookahead RIÊNG cho pure pursuit trong curve mode, tách khỏi
        # lookahead_distance (dùng cho pipeline thẳng/vision) — xem
        # curve_pp_cfg dưới đây + docstring rl_car_params.yaml.
        self.declare_parameter("curve_lookahead_distance", 1.5)
        self.declare_parameter("route_csv", "")  # rỗng = map/gps_log.csv
        self.declare_parameter("curve_course_ds_m", 0.5)
        self.declare_parameter("gps_route_stale_s", 0.5)
        # curve_zone_* dùng chung (khối /** trong yaml) để tìm curve zone đầu
        # tiên làm sân test — không liên quan gate GPS thật (_curve_mode_data_ok).
        self.declare_parameter("curve_zone_curvature_thresh", 0.05)
        # List [zone0, zone1, ...] theo thứ tự cua dọc tuyến; 1 phần tử = áp
        # chung. Xem RouteMapMatcher.detect_curve_zones.
        self.declare_parameter("curve_zone_dilate_before_m", [4.0])
        self.declare_parameter("curve_zone_dilate_after_m", [4.0])
        # --- Test bench curve mode (KHÔNG cần GPS/RouteEKF thật của gps_node,
        # nhưng VẪN dùng /odom encoder THẬT để tích phân s/d/psi_err) ---
        # true: bỏ qua toàn bộ gate GPS (in_curve_zone/sigma_d/route tươi),
        # tự khởi tạo 1 RouteEKF riêng ở đầu curve zone rồi predict() bằng
        # /odom thật mỗi tick — đẩy xe thật (tay hoặc chạy thật) để xem
        # d/heading reconstruct qua course có đúng không, không cần chờ GPS
        # fix thật hay xe thật sự đã lái vào đúng 1 zone đã map.
        self.declare_parameter("curve_frenet_force_test", False)
        self.declare_parameter("curve_frenet_test_d_m", 0.0)   # [m] d khởi tạo, panel convention +=phải
        self.declare_parameter("curve_frenet_test_psi_err_deg", 0.0)  # [deg] psi_err khởi tạo, +=phải
        # true (mặc định): lặp vô hạn trong zone — dùng khi ĐỨNG YÊN/đẩy tay,
        # chỉ xem panel, KHÔNG auto lái thật (mỗi lần lặp lại là 1 bước nhảy
        # s giật lùi ~20m, auto thật lái theo cái này sẽ "quay ngược" đột ngột
        # đúng như đã thấy). false: chạy ĐÚNG 1 LƯỢT qua zone rồi TỰ TẮT force
        # test — trả lại pipeline auto bình thường (thẳng, FrenetEKF/vision),
        # giống hệt hành vi thật lúc ra khỏi curve zone — dùng để test "bám
        # làn cua" như 1 phần của auto lái liên tục thật, không cần GPS. Muốn
        # test lại thì restart control_node (rearm zone test).
        self.declare_parameter("curve_frenet_test_loop", True)
        # --- Ghi log so sánh (d, heading) 3 nguồn ra CSV để vẽ đồ thị offline ---
        # Bật để lưu (mỗi log_compare_rate_hz, chỉ trên đoạn THẲNG — không
        # curve mode) 3 cặp (d, heading) cùng quy ước vision (d_meters +=line
        # bên phải xe, heading độ): (1) EKF fused, (2) vision thuần (đo trực
        # tiếp perception), (3) encoder thuần (1 FrenetEKF thứ 2 CHỈ predict()
        # từ /odom, seed 1 lần từ vision đầu, không bao giờ correct — cho thấy
        # trôi dead-reckon). Vẽ lại bằng scripts/plot_dheading_compare.py.
        self.declare_parameter("log_compare_enable", False)
        self.declare_parameter("log_compare_csv", "")  # rỗng = tự đặt tên theo timestamp
        self.declare_parameter("log_compare_rate_hz", 20.0)

        # --- Ghi log TIMING vòng điều khiển ra CSV (đo loop-rate/jitter THẬT) ---
        # Khác log_compare (throttle 20Hz): cái này ghi MỖI tick, không throttle,
        # đo period thật giữa 2 lần gọi cùng timer + thời gian thực thi callback.
        # 1 dòng/tick cho cả 2 timer (_control_tick 50Hz, _planner_tick 15Hz),
        # cột 'loop' phân biệt. Vẽ/thống kê bằng scripts/plot_loop_timing.py.
        # Buffer trong RAM rồi flush theo batch (log_timing_flush_n dòng) để I/O
        # đĩa không tự làm nhiễu chính phép đo timing.
        self.declare_parameter("log_timing_enable", False)
        self.declare_parameter("log_timing_csv", "")  # rỗng = ~/rl_car_timing_<stamp>.csv
        self.declare_parameter("log_timing_flush_n", 200)

        self.joy_topic = self.get_parameter("joy_topic").value
        self.r1_button_index = int(self.get_parameter("r1_button_index").value)
        self.max_linear_speed = float(self.get_parameter("max_linear_speed").value)
        self.max_angular_speed = float(self.get_parameter("max_angular_speed").value)
        self.manual_speed_axis = int(self.get_parameter("manual_speed_axis").value)
        self.manual_steer_axis = int(self.get_parameter("manual_steer_axis").value)
        self.manual_invert_speed_axis = bool(
            self.get_parameter("manual_invert_speed_axis").value
        )
        self.manual_invert_steer_axis = bool(
            self.get_parameter("manual_invert_steer_axis").value
        )
        self.manual_deadzone = float(self.get_parameter("manual_deadzone").value)
        self.auto_speed = float(self.get_parameter("auto_speed").value)

        self.pp_cfg = PurePursuitConfig(
            lookahead_distance=float(self.get_parameter("lookahead_distance").value),
            max_linear_speed=self.max_linear_speed,
            max_angular_speed=self.max_angular_speed,
            heading_gain=float(self.get_parameter("pure_pursuit_heading_gain").value),
            track_width=float(self.get_parameter("track_width").value),
            max_wheel_speed=float(self.get_parameter("max_wheel_speed").value),
        )
        # Config RIÊNG cho curve mode (_planner_tick_curve) — mọi field khác
        # giống pp_cfg, chỉ lookahead_distance khác (curve_lookahead_distance):
        # 0.8 (thẳng) cho hệ số khuếch đại lateral_error quá mạnh khi áp cho
        # curve mode (lỗi thường gặp: d lệch 0.3-1m do dead-reckon qua cua ->
        # angular_z vài rad/s -> psi vọt qua 180°/xoay tại chỗ, đo bằng mô
        # phỏng vòng kín offline) — tách riêng để tăng lookahead cho curve
        # KHÔNG ảnh hưởng pure pursuit thẳng đang chạy tốt.
        self.curve_pp_cfg = PurePursuitConfig(
            lookahead_distance=float(self.get_parameter("curve_lookahead_distance").value),
            max_linear_speed=self.max_linear_speed,
            max_angular_speed=self.max_angular_speed,
            heading_gain=self.pp_cfg.heading_gain,
            track_width=self.pp_cfg.track_width,
            max_wheel_speed=self.pp_cfg.max_wheel_speed,
        )

        self.ekf = FrenetEKF(
            q_d=float(self.get_parameter("ekf_q_d").value),
            q_psi=float(self.get_parameter("ekf_q_psi").value),
            r_d=float(self.get_parameter("ekf_r_d").value),
            r_psi=float(self.get_parameter("ekf_r_psi").value),
        )
        # EKF được đọc/ghi từ 2 timer khác nhau (_control_tick + _planner_tick,
        # 2 callback group riêng trên MultiThreadedExecutor) -> cần lock.
        self._ekf_lock = threading.Lock()

        # FrenetEKF thứ 2 CHỈ dùng cho log so sánh: predict() từ /odom mỗi tick
        # (encoder thuần), seed 1 lần từ vision đầu, KHÔNG BAO GIỜ correct() —
        # cho thấy dead-reckon encoder trôi thế nào so với EKF fused / vision.
        # Cùng tham số Q với ekf chính để so sánh công bằng. Dùng chung
        # _ekf_lock (chỉ đụng trong _control_tick predict + _frenet_state_cb seed).
        self._enc_ekf = FrenetEKF(
            q_d=float(self.get_parameter("ekf_q_d").value),
            q_psi=float(self.get_parameter("ekf_q_psi").value),
            r_d=float(self.get_parameter("ekf_r_d").value),
            r_psi=float(self.get_parameter("ekf_r_psi").value),
        )
        self._enc_seeded = False
        self._latest_vision: tuple[float, float, float] | None = None  # (d_meters, heading_deg, mono_t)

        # Mở file CSV log so sánh nếu bật (xem log_compare_* params). Chỉ
        # _control_tick ghi (1 luồng writer duy nhất) -> không cần lock file.
        self._log_writer = None
        self._log_file = None
        self._log_t0: float | None = None
        self._log_last_write = 0.0
        self.log_compare_rate_hz = max(1.0, float(self.get_parameter("log_compare_rate_hz").value))
        if bool(self.get_parameter("log_compare_enable").value):
            log_path = self.get_parameter("log_compare_csv").value
            if not log_path:
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                log_path = str(Path.home() / f"rl_car_dheading_{stamp}.csv")
            try:
                self._log_file = open(log_path, "w", newline="")
                self._log_writer = csv.writer(self._log_file)
                self._log_writer.writerow([
                    "t_s", "mode", "vision_fresh",
                    "ekf_d", "ekf_heading_deg",
                    "vision_d", "vision_heading_deg",
                    "enc_d", "enc_heading_deg",
                ])
                self.get_logger().info(f"Log so sanh (d,heading) -> {log_path}")
            except OSError as exc:
                self.get_logger().error(f"Khong mo duoc file log {log_path}: {exc}")
                self._log_writer = None

        # --- Timing log: mở file + state đo period/exec cho từng timer ---
        self._timing_writer = None
        self._timing_file = None
        self._timing_t0: float | None = None
        self._timing_buf: list[tuple] = []
        self._timing_last: dict[str, float] = {}  # loop -> mono t lần gọi trước
        self._timing_lock = threading.Lock()  # 2 timer ghi buffer từ 2 thread
        self.log_timing_flush_n = max(1, int(self.get_parameter("log_timing_flush_n").value))
        if bool(self.get_parameter("log_timing_enable").value):
            tpath = self.get_parameter("log_timing_csv").value
            if not tpath:
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                tpath = str(Path.home() / f"rl_car_timing_{stamp}.csv")
            try:
                self._timing_file = open(tpath, "w", newline="")
                self._timing_writer = csv.writer(self._timing_file)
                # loop: ten timer; period_ms: khoang cach toi lan goi TRUOC cua
                # cung timer (chu ky THAT); exec_ms: thoi gian chay callback.
                self._timing_writer.writerow(["t_s", "loop", "period_ms", "exec_ms"])
                self.get_logger().info(f"Log timing vong dieu khien -> {tpath}")
            except OSError as exc:
                self.get_logger().error(f"Khong mo duoc file timing {tpath}: {exc}")
                self._timing_writer = None

        self.planner_logic = PlannerLogic(
            plan_enable=bool(self.get_parameter("plan_enable").value),
            plan_speed=float(self.get_parameter("plan_speed").value),
            plan_robot_radius=float(self.get_parameter("plan_robot_radius").value),
            plan_road_width=float(self.get_parameter("plan_road_width").value),
            plan_max_curvature=float(self.get_parameter("plan_max_curvature").value),
            plan_lookahead=float(self.get_parameter("plan_lookahead").value),
            plan_clearance=float(self.get_parameter("plan_clearance").value),
            plan_obstacle_weight=float(self.get_parameter("plan_obstacle_weight").value),
            plan_center_offset=float(self.get_parameter("plan_center_offset").value),
            plan_center_weight=float(self.get_parameter("plan_center_weight").value),
            plan_min_horizon_s=float(self.get_parameter("plan_min_horizon_s").value),
            plan_max_horizon_s=float(self.get_parameter("plan_max_horizon_s").value),
            plan_d_road_w=float(self.get_parameter("plan_d_road_w").value),
        )

        self.gps_assist_enable = bool(self.get_parameter("gps_assist_enable").value)
        self.gps_vision_stale_s = float(self.get_parameter("gps_vision_stale_s").value)
        self.gps_max_sigma_d = float(self.get_parameter("gps_max_sigma_d").value)
        self.gps_ff_gain = float(self.get_parameter("gps_ff_gain").value)
        self.gps_d_gain = float(self.get_parameter("gps_d_gain").value)
        self.gps_force_in_curve_zone = bool(self.get_parameter("gps_force_in_curve_zone").value)
        self.gps_route_stale_s = float(self.get_parameter("gps_route_stale_s").value)
        self.planner_rate_hz = float(self.get_parameter("planner_rate_hz").value)

        self.curve_frenet_force_test = bool(self.get_parameter("curve_frenet_force_test").value)
        self.curve_frenet_test_d_m = float(self.get_parameter("curve_frenet_test_d_m").value)
        self.curve_frenet_test_psi_err = math.radians(
            float(self.get_parameter("curve_frenet_test_psi_err_deg").value)
        )
        self.curve_frenet_test_loop = bool(self.get_parameter("curve_frenet_test_loop").value)
        # True khi test_loop=false VÀ xe đã đi hết 1 lượt qua test zone — từ
        # đó _curve_mode_active() trả False dù curve_frenet_force_test vẫn
        # true, để auto rơi về pipeline bình thường (bàn giao như thật).
        self._test_pass_done = False
        self._test_zone = (0.0, 0.0)
        # RouteEKF RIÊNG cho test bench — KHÔNG phải bản của gps_node (bản đó
        # ở process khác, cần GPS fix mới initialize() được). Dựng lúc có
        # matcher (dưới), predict() bằng /odom THẬT mỗi tick — xem
        # _advance_test_state(): đẩy xe thật bằng tay/chạy thật, s/d/psi_err
        # tự tích phân từ encoder, không phải tốc độ giả lập theo thời gian.
        self._test_ekf = None

        # Curve-Frenet mode: dựng ReferenceCourse (spline CSV) 1 lần lúc init.
        # Truyền RouteMapMatcher.route_xy_local (đã smooth) vào course để s
        # của course khớp s_m mà RouteEKF publish (cùng polyline) — xem
        # docstring đầu file + ReferenceCourse (planner_motion/frenet_planner).
        # Import trong khối try để control_node vẫn chạy được khi thiếu
        # pyproj/CSV (curve mode chỉ tắt, không sập node).
        self._course = None
        self._matcher = None
        if bool(self.get_parameter("curve_frenet_enable").value) and self.gps_assist_enable:
            try:
                from ament_index_python.packages import get_package_share_directory

                from Gps.map_matcher import RouteMapMatcher
                from Gps.route_ekf import RouteEKF, RouteEKFConfig
                from planner_motion.frenet_planner import ReferenceCourse

                route_csv = self.get_parameter("route_csv").value
                csv_path = (
                    Path(route_csv)
                    if route_csv
                    else Path(get_package_share_directory("RL_CAR")) / "map" / "gps_log.csv"
                )
                # GIỮ matcher: /gps/route_state (s, d, psi_err) đo so với
                # POLYLINE của matcher này (RouteEKF cũng dựng từ cùng CSV).
                # Curve mode cần matcher.point_at/heading_at để reconstruct
                # pose (x,y,yaw) ĐÚNG frame route_state, rồi mới re-project
                # lên course (spline) — xem _planner_tick_curve.
                self._matcher = RouteMapMatcher(csv_path)
                self._course = ReferenceCourse(
                    self._matcher.route_xy_local,
                    ds=float(self.get_parameter("curve_course_ds_m").value),
                )
                self.get_logger().info(
                    f"Curve-Frenet: course {self._course.total_length_m:.0f}m "
                    f"tu {csv_path.name}"
                )

                # Sân test bench (curve_frenet_force_test): chọn curve zone
                # ĐẦU TIÊN trên tuyến làm nơi lặp test — không cần GPS thật,
                # chỉ cần hình học CSV tĩnh đã nạp sẵn ở trên.
                test_zones = self._matcher.detect_curve_zones(
                    curvature_thresh=float(
                        self.get_parameter("curve_zone_curvature_thresh").value
                    ),
                    dilate_before_m=list(
                        self.get_parameter("curve_zone_dilate_before_m").value
                    ),
                    dilate_after_m=list(
                        self.get_parameter("curve_zone_dilate_after_m").value
                    ),
                )
                self._test_zone = test_zones[0] if test_zones else (0.0, self._course.total_length_m)
                # RouteEKF riêng cho test bench: predict() bằng /odom THẬT
                # (self._latest_odom, xem _advance_test_state) — chỉ khác
                # RouteEKF thật của gps_node ở chỗ init KHÔNG cần fix GPS, mà
                # đặt cứng tại đầu curve zone với d/psi_err theo config test.
                self._test_ekf = RouteEKF(self._matcher, RouteEKFConfig(), curve_zones=test_zones)
                self._test_ekf.initialize(
                    self._test_zone[0],
                    d_m=self.curve_frenet_test_d_m,
                    psi_err=self.curve_frenet_test_psi_err,
                )
                if self.curve_frenet_force_test:
                    self.get_logger().warn(
                        f"Curve-Frenet TEST MODE dang BAT: (s,d,psi_err) tich phan tu /odom "
                        f"THAT quanh zone {self._test_zone[0]:.0f}-{self._test_zone[1]:.0f}m, "
                        f"KHONG dung GPS/RouteEKF that cua gps_node. Day xe (tay hoac chay "
                        f"that) de xem s tien theo dung quang duong. TAT curve_frenet_force_test "
                        f"truoc khi chay that."
                    )
            except Exception as exc:
                self.get_logger().warn(f"Curve-Frenet tat (khong dung duoc course): {exc}")
                self._course = None
                self._matcher = None

        self.mode = Mode.MANUAL
        self._prev_r1 = False
        self._manual_cmd = (0.0, 0.0)
        self._latest_detections: list = []
        self._latest_odom: Odometry | None = None
        self._last_predict_time = self.get_clock().now()
        self._last_vision_correct_time: float | None = None
        self._latest_gps_route: tuple[float, float, float, float, float, float] | None = None
        self._latest_gps_route_time: float | None = None
        # Profile curvature (s_grid, k_grid) từ /gps/route_state phần tử [6:]
        # (xem Gps/node.py:_curve_kappa_profile) — None nếu bag/gps_node cũ
        # chưa publish, _planner_tick rơi về kappa_ff cũ khi đó.
        self._latest_kappa_profile: tuple[np.ndarray, np.ndarray] | None = None
        # True nếu tick planner trước đó đang ở curve mode — dùng phát hiện
        # cạnh XUỐNG (ra khỏi zone) để bàn giao FrenetEKF lại cho vision.
        self._curve_mode_prev = False

        best_effort_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.cmd_vel_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.mode_pub = self.create_publisher(String, "/control/mode", 10)
        # [s, d, psi, v] của EKF — để visualization_node vẽ debug, xác nhận
        # EKF vẫn predict/correct đúng (đặc biệt lúc mất line).
        self.ekf_state_pub = self.create_publisher(
            Float64MultiArray, "/control/ekf_state", 10
        )
        # Path mà planner NỘI BỘ (EKF-driven, xem _planner_tick) vừa sinh —
        # để visualization_node vẽ đúng pipeline thật đang lái xe, không
        # phải optimal_path của planner_motion_node (chỉ dựa đo trực tiếp
        # từ camera, dùng cho viz/rosbag riêng — xem planner_motion/logic.py).
        self.planner_viz_pub = self.create_publisher(
            String, "/control/planner_viz", 10
        )

        self.create_subscription(Joy, self.joy_topic, self._joy_cb, 10)
        self.create_subscription(Odometry, "/odom", self._odom_cb, 10)
        self.create_subscription(
            String, "/perception/frenet_state", self._frenet_state_cb, best_effort_qos
        )
        if self.gps_assist_enable:
            self.create_subscription(
                Float64MultiArray, "/gps/route_state", self._gps_route_state_cb, 10
            )

        control_rate_hz = float(self.get_parameter("control_rate_hz").value)
        self.create_timer(1.0 / control_rate_hz, self._control_tick)

        # Planner nội bộ chạy callback group riêng (MutuallyExclusive) để
        # không bị chặn bởi/làm chặn _control_tick trên MultiThreadedExecutor
        # — sinh hàng trăm quỹ đạo ứng viên mỗi lần plan, có thể tốn vài chục ms.
        planner_group = MutuallyExclusiveCallbackGroup()
        planner_rate_hz = float(self.get_parameter("planner_rate_hz").value)
        self.create_timer(
            1.0 / planner_rate_hz, self._planner_tick, callback_group=planner_group
        )

        self.get_logger().info("control_node ready, mode=manual")

    def _joy_cb(self, msg: Joy) -> None:
        if len(msg.buttons) > self.r1_button_index:
            r1 = bool(msg.buttons[self.r1_button_index])
            if r1 and not self._prev_r1:
                self.mode = Mode.AUTO if self.mode is Mode.MANUAL else Mode.MANUAL
                self.get_logger().info(f"Mode -> {self.mode.value}")
            self._prev_r1 = r1

        linear_x = 0.0
        angular_z = 0.0
        if len(msg.axes) > self.manual_speed_axis:
            speed_norm = msg.axes[self.manual_speed_axis]
            if self.manual_invert_speed_axis:
                speed_norm = -speed_norm
            linear_x = _apply_deadzone(speed_norm, self.manual_deadzone) * self.max_linear_speed
        if len(msg.axes) > self.manual_steer_axis:
            steer_norm = msg.axes[self.manual_steer_axis]
            if self.manual_invert_steer_axis:
                steer_norm = -steer_norm
            angular_z = _apply_deadzone(steer_norm, self.manual_deadzone) * self.max_angular_speed
        self._manual_cmd = (linear_x, angular_z)

    def _odom_cb(self, msg: Odometry) -> None:
        self._latest_odom = msg

    def _frenet_state_cb(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        # Detections cần lấy độc lập với frenet_viz — planner nội bộ vẫn
        # cần obstacle list cả khi mất line (chỉ là sẽ thiếu phần "frenet"
        # chiếu theo lane spline, perception/logic.py tự bỏ obstacle đó).
        self._latest_detections = payload.get("detections", [])

        frenet_viz = payload.get("frenet_viz")
        if not frenet_viz:
            return
        d_meas = frenet_viz.get("d_meters_filtered")
        psi_meas_deg = frenet_viz.get("heading_filtered")
        if d_meas is None or psi_meas_deg is None:
            return

        # Lưu vision thuần cho log so sánh (luôn luôn, kể cả lúc curve/GPS mode
        # bỏ qua correct bên dưới — vẫn muốn ghi giá trị vision đo được). Seed
        # _enc_ekf 1 lần từ vision đầu tiên để encoder-only bắt đầu KHỚP vision
        # rồi dead-reckon (thấy trôi rõ). Convention EKF: d = -d_meters,
        # psi = radians(heading) -> seed reset_lateral(-d_meters, radians(hdg)).
        self._latest_vision = (float(d_meas), float(psi_meas_deg), time.monotonic())
        if self._log_writer is not None and not self._enc_seeded:
            with self._ekf_lock:
                self._enc_ekf.reset_lateral(-float(d_meas), math.radians(float(psi_meas_deg)))
            self._enc_seeded = True

        # Timestamp cập nhật TRƯỚC khi xét _use_gps_route() — camera vẫn
        # đang thấy line thật (vision không "stale"), dù có thể đang bị bỏ
        # qua bên dưới. Nhờ vậy ra khỏi cua/hết ép GPS là vision được tin
        # lại NGAY (không phải đợi thêm gps_vision_stale_s vì tưởng lâu rồi
        # chưa correct).
        self._last_vision_correct_time = time.monotonic()
        if self._use_gps_route() or self._curve_mode_active():
            # Đang chủ động tin GPS+encoder (mất line lâu HOẶC bị ép trong
            # curve zone, HOẶC curve mode reference cong đang chạy) -> vision
            # KHÔNG được sửa FrenetEKF nữa dù camera vẫn còn thấy line —
            # nếu không, vision sẽ liên tục kéo state.d/psi ngược lại trong
            # khi pure pursuit đang bám theo GPS d + kappa feed-forward
            # (hoặc route state của curve mode), 2 nguồn giằng co nhau trên
            # cùng 1 state.
            return
        with self._ekf_lock:
            self.ekf.correct(float(d_meas), float(psi_meas_deg))

    def _gps_route_state_cb(self, msg: Float64MultiArray) -> None:
        # kappa_ff (phần tử [4]) và in_curve_zone (phần tử [5]) thêm sau —
        # bag/gps_node cũ chỉ có 4, mặc định 0.0 thì feed-forward/force tắt,
        # không phải lỗi. kappa_profile (phần tử [6:] = [step_m, n, k0..kn],
        # xem Gps/node.py:_curve_kappa_profile) thêm sau nữa — bag/gps_node
        # cũ (len < 8) không có -> None, _planner_tick rơi về kappa_ff cũ.
        if len(msg.data) >= 4:
            kappa_ff = msg.data[4] if len(msg.data) >= 5 else 0.0
            in_curve_zone = msg.data[5] if len(msg.data) >= 6 else 0.0
            # (s_m, d_m, psi_err, sigma_d, kappa_ff, in_curve_zone)
            self._latest_gps_route = (*msg.data[:4], kappa_ff, in_curve_zone)
            self._latest_gps_route_time = time.monotonic()

            self._latest_kappa_profile = None
            if len(msg.data) >= 8:
                step_m = msg.data[6]
                n = int(round(msg.data[7]))
                k_vals = msg.data[8 : 8 + n + 1]
                if step_m > 0.0 and len(k_vals) == n + 1:
                    # s_grid TƯƠNG ĐỐI tính từ vị trí hiện tại (0 = xe đang ở
                    # đây) — khớp frame s0=0.0 mà FrenetOptimalPlanner.plan()
                    # dùng mỗi lần replan (planner_motion/logic.py:
                    # plan_from_state), KHÔNG phải s tuyệt đối dọc tuyến GPS.
                    s_grid = step_m * np.arange(n + 1, dtype=float)
                    self._latest_kappa_profile = (s_grid, np.array(k_vals, dtype=float))

    def _use_gps_route(self) -> bool:
        """True khi GPS đủ tin cậy (sigma_d dưới ngưỡng) VÀ (mất line đủ lâu
        HOẶC — nếu bật gps_force_in_curve_zone — đang ở trong 1 curve zone đã
        ghi nhận trên bản đồ, kể cả khi camera vẫn còn thấy line). Còn lại
        (chưa mất line, không trong zone/không bật force) thì luôn False —
        GPS không đụng vào lệch ngang khi không cần thiết. Cùng gate này cũng
        quyết định có dùng curvature feed-forward hay không (xem _planner_tick)."""
        if not self.gps_assist_enable or self._latest_gps_route is None:
            return False
        _s, _d, _psi_err, sigma_d, _kappa_ff, in_curve_zone = self._latest_gps_route
        if sigma_d >= self.gps_max_sigma_d:
            return False
        if self.gps_force_in_curve_zone and bool(in_curve_zone):
            return True
        vision_stale = (
            self._last_vision_correct_time is None
            or (time.monotonic() - self._last_vision_correct_time) > self.gps_vision_stale_s
        )
        return vision_stale

    def _curve_mode_data_ok(self) -> bool:
        """Điều kiện DỮ LIỆU tối thiểu để curve mode chạy được ở bất kỳ thời
        điểm nào (vào lẫn đang chạy): có course dựng sẵn, /gps/route_state
        đã nhận và chưa quá cũ. Không xét sigma_d/in_curve_zone — 2 cái đó
        có ý nghĩa khác nhau tuỳ đang VÀO hay đang Ở TRONG (xem
        _curve_mode_active)."""
        if self._course is None:
            return False
        if self._latest_gps_route is None or self._latest_gps_route_time is None:
            return False
        return (time.monotonic() - self._latest_gps_route_time) <= self.gps_route_stale_s

    def _vision_fresh(self) -> bool:
        """True nếu camera vừa detect line hợp lệ trong vòng gps_vision_stale_s
        giây gần nhất. self._last_vision_correct_time được cập nhật trong
        _frenet_state_cb NGAY KHI nhận được d/heading hợp lệ, TRƯỚC khi xét
        có dùng để correct FrenetEKF hay không — nên phản ánh đúng "camera
        hiện có đang thấy line" bất kể correction có bị chặn hay không (đang
        curve mode/GPS route). Dùng để GATE việc bàn giao lại vision lúc ra
        khỏi curve zone — xem _curve_mode_active()."""
        return (
            self._last_vision_correct_time is not None
            and (time.monotonic() - self._last_vision_correct_time) <= self.gps_vision_stale_s
        )

    def _curve_mode_active(self) -> bool:
        """True khi lái theo reference CONG của tuyến CSV (curve mode). Khi
        active: _planner_tick_curve thay toàn bộ pipeline thẳng — state
        (s, d, psi_err) từ RouteEKF, planner trên spline CSV, vision bị chặn
        correct (xem _frenet_state_cb).

        VÀO (cạnh lên, self._curve_mode_prev=False): gate ĐẦY ĐỦ — phải đang
        TRONG curve zone (in_curve_zone, vị trí) VÀ sigma_d đủ tin cậy
        (RouteEKF chưa dead-reckon mù quá lâu) VÀ vision ĐANG MẤT
        (not _vision_fresh()) mới cho vào. Đối xứng với gate THOÁT bên dưới
        (which requires in_curve_zone=False OR vision fresh trở lại) — cùng
        logic "curve mode chỉ chạy khi thực sự không còn vision để dùng",
        không phải chỉ dựa vị trí hình học suông. Nếu xe vào vùng cua nhưng
        camera vẫn đang thấy line (vd cua rất thoáng, chưa mất line), pipeline
        THẲNG (vision) tiếp tục lái — curve mode chỉ nhảy vào đúng lúc cả 2
        điều kiện (vị trí + mất line) cùng xảy ra.

        ĐANG CHẠY (self._curve_mode_prev=True): thoát (trả False, bàn giao
        lại vision) CHỈ KHI CẢ 2 điều kiện cùng đúng — đã ra khỏi zone theo
        VỊ TRÍ (in_curve_zone=False, có hysteresis sẵn từ RouteEKF — xem
        Gps/route_ekf.py:_update_curve_zone_state) VÀ camera đã THẤY LINE
        LẠI (_vision_fresh()). Còn TRONG zone theo vị trí, hoặc đã ra khỏi
        zone nhưng vision vẫn chưa thấy line lại (mất line ngay đúng lúc ra
        cua — vẫn khá thường xảy ra), thì TIẾP TỤC curve mode (dead-reckon
        theo course) thay vì rơi về FrenetEKF-thuần-vision trong khi vision
        chưa có gì — đúng thiết kế gốc "ra cua, camera thấy line lại thì
        tầng 1 tự re-anchor" (README), không phải chỉ dựa vị trí suông.
        KHÔNG re-check sigma_d ở đây (chỉ gate lúc VÀO) — lý do: sigma_d
        tăng đơn điệu theo THỜI GIAN trong zone (không có correction nào
        chạy khi đang curve mode — GPS bị chặn cứng, vision cũng bị chặn ở
        _frenet_state_cb) trong khi "hết cua" là chuyện VỊ TRÍ + VISION,
        2 trục độc lập với "còn tự tin không" — re-check sigma_d giữa
        chừng dễ bàn giao sớm dù xe còn giữa khúc cua.

        curve_frenet_force_test=true: bỏ qua gate GPS thật (in_curve_zone/
        sigma_d/route tươi) — True miễn có course (self._course, dựng tĩnh
        từ CSV lúc init, không phụ thuộc GPS). Nhưng KHI curve_frenet_test_loop
        =false: cũng áp CÙNG quy tắc bàn giao — chỉ tắt (trả False) khi ĐI
        xong 1 lượt qua test zone (self._test_pass_done) VÀ vision đã thấy
        line lại, không phải chỉ dựa quãng đường đã đi — test bench mô
        phỏng đúng hành vi bàn giao thật, không chỉ mô phỏng phần vị trí."""
        if self._course is None:
            return False
        if self.curve_frenet_force_test:
            test_done = self.curve_frenet_test_loop is False and self._test_pass_done
            return not (test_done and self._vision_fresh())
        if not self._curve_mode_data_ok():
            return False
        _s, _d, _psi_err, sigma_d, _kappa_ff, in_curve_zone = self._latest_gps_route
        if self._curve_mode_prev:
            return bool(in_curve_zone) or not self._vision_fresh()
        return (
            sigma_d < self.gps_max_sigma_d
            and bool(in_curve_zone)
            and not self._vision_fresh()
        )

    def _advance_test_state(self) -> tuple[float, float, float, float]:
        """Sinh (s, d, psi_err, sigma_d) cho test bench thay cho /gps/route_state
        thật — chỉ gọi khi curve_frenet_force_test=true. KHÔNG dùng tốc độ giả
        lập theo thời gian — self._test_ekf.predict() bằng v/omega THẬT từ
        /odom (self._latest_odom, cùng topic FrenetEKF/RouteEKF thật đang
        dùng) mỗi tick, y hệt cách RouteEKF thật của gps_node tích phân —
        chỉ khác là init cứng tại đầu curve zone thay vì chờ fix GPS. Đẩy xe
        thật (tay hoặc chạy thật) thì s/d/psi_err tiến theo đúng quãng đường/
        góc quay thật đo được, không phải hằng số theo dt.

        Ra khỏi curve zone (s > zone_b):
        - curve_frenet_test_loop=true (mặc định): re-initialize về đầu zone
          với d/psi_err theo cấu hình gốc — lặp vô hạn không cần can thiệp
          tay. CHỈ dùng khi xe ĐỨNG YÊN/đẩy tay quan sát panel — nếu auto
          đang lái thật liên tục, mỗi lần lặp là 1 bước s giật lùi ~20m,
          auto sẽ đột ngột đổi lệnh lái theo tham chiếu mới ("quay ngược").
        - curve_frenet_test_loop=false: KHÔNG lặp — set self._test_pass_done
          để _curve_mode_active() trả False từ tick sau, auto rơi về
          pipeline bình thường (thẳng/vision) như bàn giao thật. Dùng để
          test "bám làn cua" như 1 đoạn của auto lái LIÊN TỤC thật; test lại
          thì restart control_node để rearm."""
        dt = 1.0 / self.planner_rate_hz
        if self._latest_odom is not None:
            v = self._latest_odom.twist.twist.linear.x
            omega = self._latest_odom.twist.twist.angular.z
        else:
            v, omega = 0.0, 0.0
        self._test_ekf.predict(v_odom=v, omega_odom=omega, dt=dt)
        state = self._test_ekf.route_state()
        zone_a, zone_b = self._test_zone
        if state.s_m > zone_b:
            if self.curve_frenet_test_loop:
                self._test_ekf.initialize(
                    zone_a, d_m=self.curve_frenet_test_d_m, psi_err=self.curve_frenet_test_psi_err
                )
                state = self._test_ekf.route_state()
            else:
                self._test_pass_done = True
        return state.s_m, state.d_m, state.psi_err, state.sigma_d

    def _planner_tick_curve(self) -> None:
        """Curve mode (cấu trúc: vào cua dùng d neo vision + s neo CSV, planner
        nhận CSV bài bản qua spline, hết cua trả lại vision):
        - State (s, d, psi_err) đọc từ /gps/route_state — RouteEKF đã
          anchor_lateral(d vision) liên tục tới lúc vào zone, trong zone
          encoder predict + chiếu lên polyline CSV (s tự nhất quán hình cua).
        - Planner sinh candidate trên ReferenceCourse (spline CSV) — path
          convert sang map-local, obstacle transform từ frame xe sang map,
          max_curvature check trên path thật.
        - Pure pursuit chạy trên (s, d) tuyệt đối + feed-forward độ cong
          course tại đúng điểm lookahead. FrenetEKF hoàn toàn không tham gia.

        curve_frenet_force_test=true: (s_route, d_route, psi_route, sigma_d)
        lấy từ _advance_test_state() (RouteEKF riêng, predict() bằng /odom
        THẬT) thay vì /gps/route_state của gps_node — toàn bộ phần
        reconstruct/plan/panel bên dưới KHÔNG đổi, nên đây là bài test trung
        thực của đúng pipeline sẽ chạy thật, chỉ khác nguồn RouteEKF (không
        cần GPS fix để initialize)."""
        if self.curve_frenet_force_test:
            s_route, d_route, psi_route, sigma_d = self._advance_test_state()
        else:
            s_route, d_route, psi_route, sigma_d, _kappa_ff, _in_zone = self._latest_gps_route
        c_speed = self.planner_logic.target_speed

        # (s_route, d_route, psi_route) từ RouteEKF đo so với POLYLINE (matcher),
        # nhưng planner + feed-forward dùng SPLINE (course) — 2 tham chiếu lệch
        # tới ~7° heading ở chỗ cua gấp (đo offline). Nếu dùng thẳng route_state
        # với course thì heading/d sai đúng bằng chênh lệch đó. Sửa: reconstruct
        # pose map (x,y,yaw) theo ĐÚNG convention matcher (cái đã sinh route_state),
        # rồi RE-PROJECT lên course để (s, d, psi_err) NHẤT QUÁN với reference
        # planner dùng. Quy ước matcher: +d = phải = (sin theta, -cos theta),
        # psi_err panel (+=phải) -> world yaw CCW+ = theta_matcher - psi_err.
        mrx, mry = self._matcher.point_at(s_route)
        theta_m = self._matcher.heading_at(s_route)
        world_yaw = _wrap(theta_m - psi_route)
        car_x = float(mrx) + d_route * math.sin(theta_m)
        car_y = float(mry) - d_route * math.cos(theta_m)

        # Re-project pose lên course -> state nhất quán frame spline.
        s_now, d_now, theta = self._course.project(car_x, car_y)
        psi_now = _wrap(theta - world_yaw)
        c_d_d = c_speed * math.sin(psi_now)

        # Obstacle: (s_m = tiến trước xe, x_m = + phải xe, frame XE — xem
        # planner_motion/logic.py) -> map-local qua pose xe: forward =
        # (cos w, sin w), right = (sin w, -cos w).
        fx, fy = math.cos(world_yaw), math.sin(world_yaw)
        obstacles_xy: list[tuple[float, float]] = []
        for det in self._latest_detections:
            fr = det.get("frenet")
            if not fr or not fr.get("available"):
                continue
            s_o = float(fr.get("s_m_filtered", fr.get("s_m")))
            x_o = float(fr.get("x_m_filtered", fr.get("x_m")))
            obstacles_xy.append(
                (car_x + s_o * fx + x_o * fy, car_y + s_o * fy - x_o * fx)
            )

        best, extra = self.planner_logic.plan_on_course(
            self._course, s_now, d_now, c_d_d, obstacles_xy
        )
        if best is None:
            return

        # Feed-forward = độ cong course tại ĐÚNG điểm pure pursuit đang nhắm —
        # đổi dấu CCW+ -> panel (+=phải) tại chỗ dùng, giống Gps/node.py.
        ff_curvature = self.gps_ff_gain * (
            -self._course.kappa_ccw(s_now + self.curve_pp_cfg.lookahead_distance)
        )
        linear_x, angular_z, target_s, target_d = compute_cmd_vel(
            best.s, best.d, s_now, d_now, psi_now, self.auto_speed, self.curve_pp_cfg,
            ff_curvature=ff_curvature,
        )

        # Payload cho panel CSV-frame (visualization/logic.py:
        # draw_curve_map_panel) — mọi toạ độ là map-local của tuyến, panel vẽ
        # từ điểm đầu tới điểm cuối CSV chứ không còn ego-frame.
        theta_t = float(self._course.yaw(target_s))
        tx_ref, ty_ref = self._course.position(target_s)
        extra["curve_frame"] = True
        extra["route_s"] = float(s_now)
        extra["route_d"] = float(d_now)
        extra["route_psi_err"] = float(psi_now)
        extra["route_sigma_d"] = float(sigma_d)
        extra["car_xy"] = [car_x, car_y]
        extra["car_yaw"] = float(world_yaw)
        extra["lookahead_xy"] = [
            float(tx_ref) + target_d * math.sin(theta_t),
            float(ty_ref) - target_d * math.cos(theta_t),
        ]
        extra["obstacles_xy"] = [[float(a), float(b)] for a, b in obstacles_xy]
        extra["kappa_ff"] = ff_curvature
        extra["using_gps_route"] = True
        self.planner_viz_pub.publish(String(data=json.dumps(extra)))

        if self.mode is Mode.AUTO:
            twist = Twist()
            twist.linear.x = linear_x
            twist.angular.z = angular_z
            self.cmd_vel_pub.publish(twist)

    def _planner_tick(self) -> None:
        """1 luồng tuyến tính duy nhất, không rẽ nhánh: lấy ĐÚNG 1 snapshot
        pose EKF (predict từ /odom + correct từ perception, đã fusion sẵn ở
        _control_tick) -> Planner sinh path từ snapshot đó -> pure pursuit
        dùng ĐÚNG snapshot đó (không đọc lại EKF) + path vừa sinh -> cmd_vel.
        Chạy độc lập tần suất camera thấy line (dead-reckon qua EKF predict),
        nên mất line vẫn tiếp tục replan.

        Trong curve zone (curve_frenet_enable): rẽ sang _planner_tick_curve
        (reference cong từ CSV); ra khỏi zone thì bàn giao FrenetEKF lại cho
        vision trước khi chạy tiếp nhánh thẳng."""
        _tt = self._tick_enter("planner")
        try:
            self._planner_tick_body()
        finally:
            self._tick_exit(_tt)

    def _planner_tick_body(self) -> None:
        """Thân thật của _planner_tick — tách ra để _planner_tick bọc timing
        (try/finally) bao trọn mọi nhánh return sớm bên dưới."""
        if self._curve_mode_active():
            self._curve_mode_prev = True
            self._planner_tick_curve()
            return
        if self._curve_mode_prev:
            # Cạnh XUỐNG: vừa ra khỏi curve mode. FrenetEKF đã dead-reckon mù
            # suốt zone (vision bị chặn ở _frenet_state_cb) — d/psi của nó
            # trôi xa; nạp lại từ route state cuối (nếu còn tươi) để vision
            # correct hội tụ từ điểm hợp lý ngay tick sau.
            self._curve_mode_prev = False
            if (
                self._latest_gps_route is not None
                and self._latest_gps_route_time is not None
                and (time.monotonic() - self._latest_gps_route_time) < 2.0
            ):
                _s, d_route, psi_route, *_rest = self._latest_gps_route
                with self._ekf_lock:
                    self.ekf.reset_lateral(d_route, psi_route)
        with self._ekf_lock:
            state = self.ekf.state
        c_speed = self.planner_logic.target_speed

        d_for_plan = state.d
        # psi_for_plan LUÔN lấy từ FrenetEKF (encoder dead-reckon + camera
        # correct khi có) — GPS route CHỈ đóng góp d (lệch ngang) và kappa_ff
        # (feed-forward), không bao giờ ghi đè heading. Lý do: psi từ RouteEKF
        # đi qua 1 tầng tích phân/projection khác (xem Gps/route_ekf.py), có
        # thể trôi lệch so với psi mà FrenetEKF đang dùng để bám làn — dùng
        # chung 1 nguồn heading tránh 2 ước lượng độc lập lệch nhau.
        psi_for_plan = state.psi
        ff_curvature = 0.0
        ref_kappa = None
        kappa_profile_pts: list[list[float]] = []
        using_gps_route = self._use_gps_route()
        if using_gps_route:
            _s, gps_d, _gps_psi_err, _sigma_d, kappa_ff, _in_curve_zone = self._latest_gps_route
            # Trộn tuyến tính thay vì tin tuyệt đối: gps_d_gain=1.0 (mặc định)
            # = tin GPS hoàn toàn như cũ, 0.0 = bỏ hẳn GPS d (chỉ còn
            # FrenetEKF), giá trị giữa = tin một phần. Giảm giá trị này nếu
            # nghi ngờ RouteEKF/GPS lệch (vd encoder trôi trong cua do trượt
            # bánh làm sigma_d không phản ánh đúng độ tin cậy thực).
            d_for_plan = state.d + self.gps_d_gain * (gps_d - state.d)
            if self._latest_kappa_profile is not None:
                # Profile (s_grid, k_grid) từ /gps/route_state (xem
                # Gps/node.py:_curve_kappa_profile) — dùng để (a) bẻ cong
                # ràng buộc max_curvature của planner nội bộ theo đúng hình
                # cua thật (frenet_planner._check_paths: ref_kappa) và (b)
                # lấy feed-forward ĐÚNG tại điểm pure pursuit đang nhắm
                # (lookahead_distance) thay vì kappa_ff — 1 preview cố định
                # (curve_ff_preview_m) có thể lệch với lookahead thực tế.
                s_grid, k_grid = self._latest_kappa_profile
                ref_kappa = self._latest_kappa_profile
                ff_curvature = self.gps_ff_gain * float(
                    np.interp(self.pp_cfg.lookahead_distance, s_grid, k_grid)
                )
                kappa_profile_pts = list(zip(s_grid.tolist(), k_grid.tolist()))
            else:
                # Fallback: bag/gps_node cũ chưa publish profile.
                ff_curvature = self.gps_ff_gain * kappa_ff
        c_d_d = c_speed * math.sin(psi_for_plan)

        # Debug: in d/heading (FrenetEKF, chế độ thẳng) + x_m/z_m từng vật cản
        # detect được ra terminal (throttle để không spam ở 15Hz).
        obs_str = ", ".join(
            f"[{det.get('label', '?')} x={fr.get('x_m_filtered', fr.get('x_m')):.2f}m "
            f"z={fr.get('z_m_filtered', fr.get('z_m')):.2f}m]"
            for det in self._latest_detections
            if (fr := det.get("frenet")) and fr.get("available")
        ) or "(khong co vat can)"
        self.get_logger().info(
            f"[THANG] d={d_for_plan:+.3f}m heading={math.degrees(psi_for_plan):+.1f}deg  vat_can: {obs_str}",
            throttle_duration_sec=0.5,
        )

        best, extra = self.planner_logic.plan_from_state(
            d_for_plan, c_d_d, self._latest_detections, ref_kappa=ref_kappa
        )
        if best is None:
            return

        # path (best.s/best.d) được sinh từ đúng d_for_plan ở trên -> pose
        # "hiện tại" truyền vào pure pursuit PHẢI cùng nguồn đó (d_for_plan,
        # có thể khác state.d khi dùng GPS), nếu không lateral_error sẽ lệch
        # giả tạo giữa 2 nguồn khác nhau. psi_for_plan luôn = state.psi (GPS
        # không ghi đè heading) nên không có vấn đề lệch nguồn ở psi.
        # s_now = 0.0: plan_from_state() luôn sinh path với s0=0.0 (vị trí xe
        # NGAY LÚC plan chạy) nên xe luôn ở s=0 trên path vừa sinh.
        linear_x, angular_z, target_s, target_d = compute_cmd_vel(
            best.s, best.d, 0.0, d_for_plan, psi_for_plan, self.auto_speed, self.pp_cfg,
            ff_curvature=ff_curvature,
        )

        # Payload theo đúng format mà OverlayRenderer.draw_frenet_panel đang
        # đọc (visualization/logic.py) — d_meters_filtered đổi dấu lại về
        # quy ước "perception" (+d = line bên phải xe) vì draw_frenet_panel
        # tự đổi dấu 1 lần khi vẽ vehicle marker; optimal_path/candidate_paths/
        # lookahead_point đã đúng quy ước panel (d, s) sẵn, không cần đổi.
        extra["d_meters_filtered"] = -d_for_plan
        extra["heading_filtered"] = math.degrees(psi_for_plan)
        extra["s_max_m"] = max(float(best.s.max()) * 1.15 if len(best.s) else 0.5, 0.5)
        extra["lookahead_point"] = [target_d, target_s]
        # kappa_ff (panel convention, +=phải): visualization_node dùng để uốn
        # cong path vẽ ra (draw_frenet_panel) khớp đúng hình dạng cua thật khi
        # đang lái bằng GPS+encoder — 0.0 khi không dùng GPS route.
        extra["kappa_ff"] = ff_curvature
        # True khi _use_gps_route() đang active (xem hàm đó) — kể cả ngoài
        # curve zone (kappa_ff=0 lúc đó) vẫn có thể True nếu chỉ mất line,
        # nên tách riêng cờ này thay vì suy từ kappa_ff!=0 — visualization_node
        # dùng để báo rõ đang ở mode GPS+encoder hay không.
        extra["using_gps_route"] = using_gps_route
        # [[s_i, kappa_i], ...] (rỗng nếu không dùng GPS route hoặc bag cũ
        # chưa có profile) — visualization_node dùng để uốn panel Frenet
        # khớp đúng hình cua thật thay vì cung tròn hằng số (xem
        # visualization/logic.py: draw_frenet_panel/bend()).
        extra["kappa_profile"] = kappa_profile_pts
        self.planner_viz_pub.publish(String(data=json.dumps(extra)))

        if self.mode is Mode.AUTO:
            twist = Twist()
            twist.linear.x = linear_x
            twist.angular.z = angular_z
            self.cmd_vel_pub.publish(twist)

    def _tick_enter(self, loop: str) -> tuple[str, float, float | None] | None:
        """Mở phép đo timing của 1 callback: lấy thời điểm vào + chu kỳ THẬT so
        với lần gọi TRƯỚC của CÙNG timer. Trả token (loop, t_enter, prev) để
        truyền cho _tick_exit() ở cuối callback — token là biến LOCAL nên 2
        timer chạy song song (2 thread, callback group riêng) không giẫm nhau.
        Trả None nếu timing tắt (callback bỏ qua đo)."""
        if self._timing_writer is None:
            return None
        t_enter = time.monotonic()
        prev = self._timing_last.get(loop)  # đọc/ghi key riêng theo loop, không đua chéo
        self._timing_last[loop] = t_enter
        return loop, t_enter, prev

    def _tick_exit(self, token: tuple[str, float, float | None] | None) -> None:
        """Đóng phép đo: tính exec_ms + period_ms, đẩy vào buffer (list.append
        atomic dưới GIL), flush theo batch dưới _timing_lock để 2 thread không
        double-flush."""
        if token is None:
            return
        loop, t_enter, prev = token
        t_exit = time.monotonic()
        with self._timing_lock:
            if self._timing_t0 is None:
                self._timing_t0 = t_enter
            period_ms = "" if prev is None else f"{(t_enter - prev) * 1000:.3f}"
            exec_ms = f"{(t_exit - t_enter) * 1000:.3f}"
            self._timing_buf.append(
                (f"{t_enter - self._timing_t0:.4f}", loop, period_ms, exec_ms)
            )
            if len(self._timing_buf) >= self.log_timing_flush_n:
                self._timing_writer.writerows(self._timing_buf)
                self._timing_buf.clear()
                self._timing_file.flush()

    def _control_tick(self) -> None:
        """EKF fusion (predict mỗi tick từ /odom) + publish mode/manual — auto
        cmd_vel publish từ _planner_tick (xem ở trên), không lặp lại ở đây."""
        _tt = self._tick_enter("control")
        now = self.get_clock().now()
        if self._latest_odom is not None:
            dt = (now - self._last_predict_time).nanoseconds * 1e-9
            self._last_predict_time = now
            with self._ekf_lock:
                self.ekf.predict(
                    self._latest_odom.twist.twist.linear.x,
                    self._latest_odom.twist.twist.angular.z,
                    dt,
                )
                # encoder-only EKF: cùng predict, KHÔNG correct (xem __init__).
                self._enc_ekf.predict(
                    self._latest_odom.twist.twist.linear.x,
                    self._latest_odom.twist.twist.angular.z,
                    dt,
                )

        self.mode_pub.publish(String(data=self.mode.value))
        with self._ekf_lock:
            ekf_state = self.ekf.state
        self.ekf_state_pub.publish(
            Float64MultiArray(data=[ekf_state.d, ekf_state.psi])
        )

        self._maybe_log_compare()

        if self.mode is Mode.MANUAL:
            twist = Twist()
            twist.linear.x, twist.angular.z = self._manual_cmd
            self.cmd_vel_pub.publish(twist)

        self._tick_exit(_tt)

    def _maybe_log_compare(self) -> None:
        """Ghi 1 dòng CSV so sánh (d, heading) 3 nguồn — chỉ khi bật log VÀ
        đang ở đoạn THẲNG (không curve mode), throttle theo log_compare_rate_hz.
        Mọi cột (d, heading) cùng quy ước VISION: d = d_meters (+d = line/lệch
        sang phải theo perception), heading = độ. EKF/encoder lưu d dạng path
        (= -d_meters) nên đổi dấu về vision (-state.d) khi ghi."""
        if self._log_writer is None or self._curve_mode_active():
            return
        now = time.monotonic()
        if (now - self._log_last_write) < (1.0 / self.log_compare_rate_hz):
            return
        self._log_last_write = now
        if self._log_t0 is None:
            self._log_t0 = now

        with self._ekf_lock:
            st = self.ekf.state
            en = self._enc_ekf.state
        ekf_d = -st.d
        ekf_hdg = math.degrees(st.psi)
        enc_d = -en.d
        enc_hdg = math.degrees(en.psi)

        vis = self._latest_vision
        if vis is not None and (now - vis[2]) <= self.gps_vision_stale_s:
            vision_d, vision_hdg, vision_fresh = vis[0], vis[1], 1
        elif vis is not None:
            vision_d, vision_hdg, vision_fresh = vis[0], vis[1], 0  # giá trị cũ, cờ 0
        else:
            vision_d = vision_hdg = float("nan")
            vision_fresh = 0

        self._log_writer.writerow([
            f"{now - self._log_t0:.3f}", self.mode.value, vision_fresh,
            f"{ekf_d:.4f}", f"{ekf_hdg:.3f}",
            f"{vision_d:.4f}", f"{vision_hdg:.3f}",
            f"{enc_d:.4f}", f"{enc_hdg:.3f}",
        ])
        self._log_file.flush()

    def destroy_node(self) -> bool:
        if self._log_file is not None:
            try:
                self._log_file.close()
            except OSError:
                pass
        # Flush nốt buffer timing còn dở rồi đóng file.
        if self._timing_writer is not None:
            try:
                with self._timing_lock:
                    if self._timing_buf:
                        self._timing_writer.writerows(self._timing_buf)
                        self._timing_buf.clear()
                self._timing_file.close()
            except OSError:
                pass
        return super().destroy_node()


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = ControlNode()
    executor = MultiThreadedExecutor(num_threads=3)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
