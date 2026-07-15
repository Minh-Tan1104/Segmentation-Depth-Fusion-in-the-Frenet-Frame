"""ROS2 node: GPS chỉ dẫn đường (route s/d/psi_err), KHÔNG điều khiển xe.

Vai trò (xem thảo luận thiết kế trong lịch sử session — GPS không bao giờ lái
xe trực tiếp):
  - Đọc UBX từ serial (gps_reader.extract_sample), cùng logic với Gps.py.
  - RouteEKF (route_ekf.py) chạy NỀN liên tục: predict từ /odom mỗi tick,
    correct từ GPS khi có fix đạt chất lượng (R = hAcc^2 trung thực).
  - anchor_lateral() liên tục từ /control/ekf_state (FrenetEKF của
    control_node, đã correct bằng camera) MỖI KHI vision còn "tươi" (vừa nhận
    /perception/frenet_state hợp lệ trong vòng vision_timeout_s) — nhờ vậy
    lúc vision mất (mất line ở ngã rẽ), RouteEKF đã có sẵn d/psi_err khớp với
    thực tế, không phải đoán mù từ đầu.
  - Publish /gps/route_state = [s_m, d_m, psi_err, sigma_d, kappa_ff,
    in_curve_zone] để control_node TỰ QUYẾT có dùng hay không (control_node
    giữ toàn quyền lái — xem control/node.py, tham số gps_assist_enable).
  - kappa_ff (phần tử thứ 5): độ cong CÓ DẤU của tuyến CSV tại
    s + curve_ff_preview_m, CHỈ khác 0 khi điểm preview đó rơi vào 1 curve
    zone (matcher.detect_curve_zones(), tính 1 lần lúc init — xem __init__).
    Ngoài zone luôn ép về 0.0, kể cả khi κ hình học khác 0 chút ít (nhiễu ghi
    CSV) — tránh feed-forward "rung" tay lái trên đoạn gần thẳng. Dấu: +
    = cua PHẢI (quy ước panel, khớp d_m/psi_err), đổi dấu tại nguồn vì
    matcher.curvature_at() trả CCW+ = cua trái (toán học chuẩn).
    control_node (pure_pursuit) cộng kappa_ff làm feed-forward, chỉ khi
    _use_gps_route() — xem control/node.py, control/pure_pursuit.py.
  - in_curve_zone (phần tử thứ 6, 1.0/0.0): true nếu vị trí HIỆN TẠI (không
    preview) đang trong 1 curve zone. Dùng cho gps_force_in_curve_zone ở
    control_node — CHỦ ĐỘNG chuyển sang GPS+encoder ngay khi vào cua, không
    cần chờ mất line như đường _use_gps_route() mặc định.

RouteEKF chưa initialize() cho tới khi có fix GPS đầu tiên đạt chất lượng
(không biết xe đang ở đâu trên tuyến trước đó) — trước lúc đó không publish
gì cả, control_node phải tự chịu bằng dead-reckoning FrenetEKF như cũ.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Float64MultiArray, String
from nav_msgs.msg import Odometry

from .gps_reader import extract_sample
from .map_matcher import RouteMapMatcher
from .route_ekf import RouteEKF, RouteEKFConfig

try:
    from pyubx2 import UBXReader
    from serial import Serial, SerialException
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Missing dependency '%s'. Install with: pip install pyserial pyubx2"
        % (exc.name or "unknown")
    ) from exc


class GpsNode(Node):
    def __init__(self) -> None:
        super().__init__("gps_node")

        self.declare_parameter("serial_port", "/dev/ttyUSB0")
        self.declare_parameter("baud_rate", 230400)
        self.declare_parameter("serial_timeout", 1.0)
        self.declare_parameter("route_csv", "")  # rỗng = default trong map_matcher
        self.declare_parameter("predict_rate_hz", 50.0)
        self.declare_parameter("vision_timeout_s", 0.3)
        self.declare_parameter("odom_topic", "/odom")
        self.declare_parameter("ekf_state_topic", "/control/ekf_state")
        self.declare_parameter("frenet_state_topic", "/perception/frenet_state")
        self.declare_parameter("route_state_topic", "/gps/route_state")
        # RouteEKFConfig — xem Gps/route_ekf.py, đã tune bằng sim_route_ekf.py.
        self.declare_parameter("q_xy", 0.01)
        self.declare_parameter("q_psi", 2.0e-4)
        self.declare_parameter("min_fix_type", 3)
        self.declare_parameter("max_h_acc_m", 5.0)
        self.declare_parameter("default_h_acc_m", 3.0)
        self.declare_parameter("gps_gate_chi2", 9.21)
        self.declare_parameter("anchor_sigma_d", 0.15)
        self.declare_parameter("project_window_m", 40.0)
        self.declare_parameter("curve_gps_r_scale", 4.0)
        # --- Curvature feed-forward (chỉ dùng trong curve zone, xem docstring) ---
        self.declare_parameter("curve_ff_preview_m", 1.5)
        self.declare_parameter("curve_zone_curvature_thresh", 0.05)
        self.declare_parameter("curve_zone_dilate_m", 4.0)

        route_csv = self.get_parameter("route_csv").value
        if route_csv:
            csv_path = Path(route_csv)
        else:
            # Cài qua colcon thì node.py nằm trong install/, không phải source
            # tree -> phải lấy map/ qua share dir (đã đăng ký trong setup.py),
            # không suy từ __file__ như Gps.py (chạy trực tiếp từ source).
            from ament_index_python.packages import get_package_share_directory

            csv_path = Path(get_package_share_directory("RL_CAR")) / "map" / "gps_path_2m.csv"
        self.matcher = RouteMapMatcher(csv_path)

        # Curve zones tính 1 lần từ hình học CSV lúc load (tuyến không đổi khi
        # đang chạy) — dùng để gate curvature feed-forward trong _tick(), xem
        # docstring đầu file. Không liên quan gate map-matching (update()).
        self.curve_ff_preview_m = float(self.get_parameter("curve_ff_preview_m").value)
        self._curve_zones = self.matcher.detect_curve_zones(
            curvature_thresh=float(self.get_parameter("curve_zone_curvature_thresh").value),
            dilate_m=float(self.get_parameter("curve_zone_dilate_m").value),
        )
        self.get_logger().info(
            f"Curve feed-forward: {len(self._curve_zones)} zone tren tuyen "
            f"({', '.join(f'{a:.0f}-{b:.0f}m' for a, b in self._curve_zones)})"
        )

        self.route_ekf = RouteEKF(
            self.matcher,
            RouteEKFConfig(
                q_xy=float(self.get_parameter("q_xy").value),
                q_psi=float(self.get_parameter("q_psi").value),
                min_fix_type=int(self.get_parameter("min_fix_type").value),
                max_h_acc_m=float(self.get_parameter("max_h_acc_m").value),
                default_h_acc_m=float(self.get_parameter("default_h_acc_m").value),
                gps_gate_chi2=float(self.get_parameter("gps_gate_chi2").value),
                anchor_sigma_d=float(self.get_parameter("anchor_sigma_d").value),
                project_window_m=float(self.get_parameter("project_window_m").value),
                curve_gps_r_scale=float(self.get_parameter("curve_gps_r_scale").value),
            ),
            curve_zones=self._curve_zones,
        )
        self._ekf_lock = threading.Lock()

        self.vision_timeout_s = float(self.get_parameter("vision_timeout_s").value)
        self._latest_odom: Odometry | None = None
        self._last_predict_time = self.get_clock().now()
        self._last_vision_fresh_time: float | None = None
        self._latest_camera_ekf_state: tuple[float, float, float, float] | None = None

        best_effort_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.route_state_pub = self.create_publisher(
            Float64MultiArray, self.get_parameter("route_state_topic").value, 10
        )

        self.create_subscription(
            Odometry, self.get_parameter("odom_topic").value, self._odom_cb, 10
        )
        self.create_subscription(
            Float64MultiArray,
            self.get_parameter("ekf_state_topic").value,
            self._camera_ekf_state_cb,
            10,
        )
        self.create_subscription(
            String,
            self.get_parameter("frenet_state_topic").value,
            self._frenet_state_cb,
            best_effort_qos,
        )

        predict_rate_hz = float(self.get_parameter("predict_rate_hz").value)
        self.create_timer(1.0 / predict_rate_hz, self._tick)

        # Đọc serial trên thread riêng (UBXReader.read() blocking) — không
        # được chặn executor của ROS, giống lý do Encoder/node.py dùng
        # non-blocking poll, nhưng ở đây đơn giản hơn bằng 1 thread độc lập
        # cộng _ekf_lock để đồng bộ với _tick().
        port = self.get_parameter("serial_port").value
        baud = int(self.get_parameter("baud_rate").value)
        timeout = float(self.get_parameter("serial_timeout").value)
        try:
            self._serial = Serial(port, baud, timeout=timeout)
        except SerialException as exc:
            self.get_logger().error(f"Khong mo duoc serial GPS {port}: {exc}")
            self._serial = None

        self._stop_event = threading.Event()
        self._reader_thread: threading.Thread | None = None
        if self._serial is not None:
            self._reader_thread = threading.Thread(
                target=self._serial_loop, daemon=True
            )
            self._reader_thread.start()

        self.get_logger().info(
            f"gps_node ready, route={self.matcher.total_length_m:.0f}m "
            f"{len(self.matcher.route_lat)} points, cho fix GPS dau tien de initialize"
        )

    # ------------------------------------------------------------------ #

    def _odom_cb(self, msg: Odometry) -> None:
        self._latest_odom = msg

    def _camera_ekf_state_cb(self, msg: Float64MultiArray) -> None:
        if len(msg.data) >= 4:
            self._latest_camera_ekf_state = tuple(msg.data[:4])

    def _frenet_state_cb(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        frenet_viz = payload.get("frenet_viz")
        if not frenet_viz:
            return
        if frenet_viz.get("d_meters_filtered") is None:
            return
        self._last_vision_fresh_time = time.monotonic()

    def _serial_loop(self) -> None:
        reader = UBXReader(self._serial)
        while not self._stop_event.is_set():
            try:
                _, msg = reader.read()
            except Exception as exc:  # noqa: BLE001 - serial/parse lỗi thoáng qua, log rồi thử tiếp
                self.get_logger().warn(f"Loi doc GPS serial: {exc}")
                continue
            if msg is None:
                # Timeout doc serial — khong nhan duoc byte nao (mat ket noi
                # module GPS hoac day serial sai). Throttle vi vong lap nay
                # chay lien tuc theo serial_timeout.
                self.get_logger().warn(
                    "GPS: khong nhan duoc du lieu tu serial (mat ket noi hoac "
                    "chua co tin hieu)...",
                    throttle_duration_sec=2.0,
                )
                continue
            sample = extract_sample(msg)
            if sample is None:
                # Co byte/message nhung khong phai fix hop le (vd chua co
                # fix, hoac message khac NAV-PVT) — khac voi "mat ket noi" o
                # tren, o day module GPS van song, chi la chua dinh vi duoc.
                self.get_logger().info(
                    "GPS: co du lieu nhung chua co fix hop le...",
                    throttle_duration_sec=2.0,
                )
                continue

            with self._ekf_lock:
                if not self.route_ekf.initialized:
                    if (
                        sample.fix_type is not None
                        and sample.fix_type >= self.route_ekf.config.min_fix_type
                        and sample.h_acc_m is not None
                        and sample.h_acc_m <= self.route_ekf.config.max_h_acc_m
                    ):
                        x, y = self.matcher.to_local(sample.lat, sample.lon)
                        s0, d0, _ = self.matcher.project_xy(x, y)
                        self.route_ekf.initialize(s0, d0, psi_err=0.0)
                        self.get_logger().info(
                            f"RouteEKF initialized: s={s0:.1f}m d={d0:.1f}m "
                            f"(fixType={sample.fix_type}, hAcc={sample.h_acc_m:.1f}m)"
                        )
                    else:
                        self.get_logger().info(
                            f"GPS: co fix nhung chua du chat luong de init "
                            f"(fixType={sample.fix_type}, hAcc="
                            f"{'?' if sample.h_acc_m is None else f'{sample.h_acc_m:.1f}m'}, "
                            f"can fixType>={self.route_ekf.config.min_fix_type} "
                            f"va hAcc<={self.route_ekf.config.max_h_acc_m}m)",
                            throttle_duration_sec=2.0,
                        )
                else:
                    ok, reason = self.route_ekf.correct_gps(
                        sample.lat, sample.lon, sample.h_acc_m, sample.fix_type
                    )
                    if ok:
                        self.get_logger().info(
                            f"GPS: fix OK sats={sample.satellites} "
                            f"hAcc={sample.h_acc_m:.1f}m",
                            throttle_duration_sec=5.0,
                        )
                    else:
                        self.get_logger().info(
                            f"GPS: fix bi loai ({reason})",
                            throttle_duration_sec=2.0,
                        )

    def _tick(self) -> None:
        with self._ekf_lock:
            if not self.route_ekf.initialized:
                return

            now_ros = self.get_clock().now()
            if self._latest_odom is not None:
                dt = (now_ros - self._last_predict_time).nanoseconds * 1e-9
                self._last_predict_time = now_ros
                self.route_ekf.predict(
                    self._latest_odom.twist.twist.linear.x,
                    self._latest_odom.twist.twist.angular.z,
                    dt,
                )
            else:
                self._last_predict_time = now_ros

            vision_fresh = (
                self._last_vision_fresh_time is not None
                and (time.monotonic() - self._last_vision_fresh_time) < self.vision_timeout_s
            )
            if vision_fresh and self._latest_camera_ekf_state is not None:
                _s, d_cam, psi_cam, _v = self._latest_camera_ekf_state
                self.route_ekf.anchor_lateral(d_m=d_cam, psi_err=psi_cam)

            state = self.route_ekf.route_state()

        kappa_ff = self._curve_ff_kappa(state.s_m)
        in_curve_zone = 1.0 if self._in_curve_zone(state.s_m) else 0.0
        self.route_state_pub.publish(
            Float64MultiArray(
                data=[state.s_m, state.d_m, state.psi_err, state.sigma_d, kappa_ff, in_curve_zone]
            )
        )

    def _in_curve_zone(self, s_m: float) -> bool:
        """True nếu vị trí HIỆN TẠI (không phải điểm preview) rơi vào 1 curve
        zone đã dò trên CSV — dùng cho control_node quyết định có ép chuyển
        sang GPS+encoder ngay khi vào cua hay không (gps_force_in_curve_zone),
        khác với _curve_ff_kappa() (dùng điểm preview để lái mượt hơn)."""
        return any(a <= s_m <= b for a, b in self._curve_zones)

    def _curve_ff_kappa(self, s_m: float) -> float:
        """Độ cong feed-forward tại s_m + preview, quy ước panel (+ = phải).

        0.0 nếu điểm preview KHÔNG rơi vào curve zone nào — ngoài zone hình
        học tuyến gần thẳng, curvature_at() chỉ trả nhiễu ghi CSV, không nên
        cho nó vào tay lái (xem docstring đầu file)."""
        s_preview = s_m + self.curve_ff_preview_m
        if not any(a <= s_preview <= b for a, b in self._curve_zones):
            return 0.0
        return -self.matcher.curvature_at(s_preview)

    def destroy_node(self) -> bool:
        self._stop_event.set()
        if self._reader_thread is not None:
            self._reader_thread.join(timeout=2.0)
        if self._serial is not None:
            try:
                self._serial.close()
            except Exception:
                pass
        return super().destroy_node()


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = GpsNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
