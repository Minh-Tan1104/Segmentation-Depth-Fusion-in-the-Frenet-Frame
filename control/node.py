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
"""
from __future__ import annotations

import enum
import json
import math
import threading
import time

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
        self.declare_parameter("ekf_q_s", 0.02)
        self.declare_parameter("ekf_q_d", 0.01)
        self.declare_parameter("ekf_q_psi", 0.01)
        self.declare_parameter("ekf_q_v", 0.05)
        self.declare_parameter("ekf_r_d", 0.04)
        self.declare_parameter("ekf_r_psi", math.radians(5.0) ** 2)
        self.declare_parameter("ekf_r_v", 0.02)
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

        self.ekf = FrenetEKF(
            q_s=float(self.get_parameter("ekf_q_s").value),
            q_d=float(self.get_parameter("ekf_q_d").value),
            q_psi=float(self.get_parameter("ekf_q_psi").value),
            q_v=float(self.get_parameter("ekf_q_v").value),
            r_d=float(self.get_parameter("ekf_r_d").value),
            r_psi=float(self.get_parameter("ekf_r_psi").value),
            r_v=float(self.get_parameter("ekf_r_v").value),
        )
        # EKF được đọc/ghi từ 2 timer khác nhau (_control_tick + _planner_tick,
        # 2 callback group riêng trên MultiThreadedExecutor) -> cần lock.
        self._ekf_lock = threading.Lock()

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

        self.mode = Mode.MANUAL
        self._prev_r1 = False
        self._manual_cmd = (0.0, 0.0)
        self._latest_detections: list = []
        self._latest_odom: Odometry | None = None
        self._last_predict_time = self.get_clock().now()
        self._last_vision_correct_time: float | None = None
        self._latest_gps_route: tuple[float, float, float, float, float, float] | None = None

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
        with self._ekf_lock:
            self.ekf.correct(float(d_meas), float(psi_meas_deg))
        self._last_vision_correct_time = time.monotonic()

    def _gps_route_state_cb(self, msg: Float64MultiArray) -> None:
        # kappa_ff (phần tử [4]) và in_curve_zone (phần tử [5]) thêm sau —
        # bag/gps_node cũ chỉ có 4, mặc định 0.0 thì feed-forward/force tắt,
        # không phải lỗi.
        if len(msg.data) >= 4:
            kappa_ff = msg.data[4] if len(msg.data) >= 5 else 0.0
            in_curve_zone = msg.data[5] if len(msg.data) >= 6 else 0.0
            # (s_m, d_m, psi_err, sigma_d, kappa_ff, in_curve_zone)
            self._latest_gps_route = (*msg.data[:4], kappa_ff, in_curve_zone)

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

    def _planner_tick(self) -> None:
        """1 luồng tuyến tính duy nhất, không rẽ nhánh: lấy ĐÚNG 1 snapshot
        pose EKF (predict từ /odom + correct từ perception, đã fusion sẵn ở
        _control_tick) -> Planner sinh path từ snapshot đó -> pure pursuit
        dùng ĐÚNG snapshot đó (không đọc lại EKF) + path vừa sinh -> cmd_vel.
        Chạy độc lập tần suất camera thấy line (dead-reckon qua EKF predict),
        nên mất line vẫn tiếp tục replan."""
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
        using_gps_route = self._use_gps_route()
        if using_gps_route:
            _s, gps_d, _gps_psi_err, _sigma_d, kappa_ff, _in_curve_zone = self._latest_gps_route
            # Trộn tuyến tính thay vì tin tuyệt đối: gps_d_gain=1.0 (mặc định)
            # = tin GPS hoàn toàn như cũ, 0.0 = bỏ hẳn GPS d (chỉ còn
            # FrenetEKF), giá trị giữa = tin một phần. Giảm giá trị này nếu
            # nghi ngờ RouteEKF/GPS lệch (vd encoder trôi trong cua do trượt
            # bánh làm sigma_d không phản ánh đúng độ tin cậy thực).
            d_for_plan = state.d + self.gps_d_gain * (gps_d - state.d)
            ff_curvature = self.gps_ff_gain * kappa_ff
        c_d_d = c_speed * math.sin(psi_for_plan)

        best, extra = self.planner_logic.plan_from_state(
            d_for_plan, c_d_d, self._latest_detections
        )
        if best is None:
            return

        # path (best.s/best.d) được sinh từ đúng d_for_plan ở trên -> pose
        # "hiện tại" truyền vào pure pursuit PHẢI cùng nguồn đó (d_for_plan,
        # có thể khác state.d khi dùng GPS), nếu không lateral_error sẽ lệch
        # giả tạo giữa 2 nguồn khác nhau. psi_for_plan luôn = state.psi (GPS
        # không ghi đè heading) nên không có vấn đề lệch nguồn ở psi.
        # state.s (quãng đường dead-reckon từ lần replan trước) không có
        # tương đương bên GPS nên vẫn giữ nguyên từ FrenetEKF ở cả 2 chế độ.
        linear_x, angular_z, target_s, target_d = compute_cmd_vel(
            best.s, best.d, state.s, d_for_plan, psi_for_plan, self.auto_speed, self.pp_cfg,
            ff_curvature=ff_curvature,
        )

        with self._ekf_lock:
            self.ekf.reset_s_origin()

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
        self.planner_viz_pub.publish(String(data=json.dumps(extra)))

        if self.mode is Mode.AUTO:
            twist = Twist()
            twist.linear.x = linear_x
            twist.angular.z = angular_z
            self.cmd_vel_pub.publish(twist)

    def _control_tick(self) -> None:
        """EKF fusion (predict mỗi tick từ /odom) + publish mode/manual — auto
        cmd_vel publish từ _planner_tick (xem ở trên), không lặp lại ở đây."""
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

        self.mode_pub.publish(String(data=self.mode.value))
        with self._ekf_lock:
            ekf_state = self.ekf.state
        self.ekf_state_pub.publish(
            Float64MultiArray(data=[ekf_state.s, ekf_state.d, ekf_state.psi, ekf_state.v])
        )

        if self.mode is Mode.MANUAL:
            twist = Twist()
            twist.linear.x, twist.angular.z = self._manual_cmd
            self.cmd_vel_pub.publish(twist)


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
