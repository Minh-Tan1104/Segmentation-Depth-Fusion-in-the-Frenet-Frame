"""ROS2 node: GPS chỉ dẫn đường (route s/d/psi_err), KHÔNG điều khiển xe.

Vai trò (xem thảo luận thiết kế trong lịch sử session — GPS không bao giờ lái
xe trực tiếp):
  - Đọc UBX từ serial (gps_reader.extract_sample), cùng logic với Gps.py.
  - RouteEKF (route_ekf.py) chạy NỀN liên tục: predict từ /odom mỗi tick,
    correct từ GPS khi có fix đạt chất lượng (R = hAcc^2 trung thực).
  - VISION KHÔNG BAO GIỜ ĐỤNG VÀO STATE khi RouteEKF đang chạy bình thường.
    Tầng 2 là GPS + encoder THUẦN: predict() từ /odom, correct_gps() từ fix.
    /control/ekf_state (FrenetEKF, đã correct bằng camera) chỉ được GHI ĐỆM
    lại (_last_vision_anchor) chứ không correct gì cả.
    Mẫu vision đệm đó CHỈ dùng ĐÚNG 1 LẦN, tại cạnh lên của curve zone
    (ngoài -> trong), làm ĐIỀU KIỆN ĐẦU cho đoạn cua: anchor_lateral() chốt
    d/psi_err rồi thôi, suốt phần còn lại của cua là encoder dead-reckon.
    Lý do bỏ hành vi cũ (anchor mỗi tick 50 Hz khi ngoài zone):
    anchor_lateral() là phép GÁN ĐÈ state + reset P, không qua Kalman/gate
    nào — một mẫu segmentation nhảy làn khiến pose RouteEKF dịch ngang tức
    thời (đo được 2.95 m trong 1 mẫu, trong khi encoder chỉ đi 0.088 m và
    GPS đứng yên), đồng thời P bị reset mỗi tick nên sigma_d báo 0.16 m
    "rất chắc chắn" ngay tại chỗ đang nhảy 3 m — con số giả đó lọt qua gate
    gps_max_sigma_d của control_node. Xem anchor_on_curve_entry_only.
    Cách này cũng khớp đúng với Gps/sim_route_ekf.py (vốn chỉ neo tại mép
    cua) — trước đây sim và node thật lệch nhau nên sim luôn PASS.
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
  - Từ phần tử thứ 7 trở đi: profile curvature [step_m, n, kappa_0..kappa_n]
    (xem _curve_kappa_profile()) — kappa_i là độ cong (quy ước dấu panel,
    khớp kappa_ff) tại s_m + i*step_m, cùng gate curve-zone như kappa_ff.
    control_node dùng profile này để (a) bẻ cong ràng buộc max_curvature của
    planner nội bộ theo đúng hình cua thật thay vì đường thẳng
    (planner_motion/frenet_planner.py: ref_kappa), và (b) nội suy feed-forward
    ĐÚNG tại điểm pure pursuit đang nhắm (lookahead_distance) thay vì 1 giá
    trị preview cố định (curve_ff_preview_m) có thể lệch với lookahead thực
    tế. visualization_node cũng dùng profile để uốn panel Frenet khớp hình
    cua thật (xem visualization/logic.py: draw_frenet_panel/bend()). Consumer
    cũ chỉ đọc 6 phần tử đầu vẫn hoạt động bình thường (backward compatible).

RouteEKF chưa initialize() cho tới khi có fix GPS đầu tiên đạt chất lượng
(không biết xe đang ở đâu trên tuyến trước đó) — trước lúc đó không publish
gì cả, control_node phải tự chịu bằng dead-reckoning FrenetEKF như cũ.
"""
from __future__ import annotations

import csv
import json
import math
import threading
import time
from datetime import datetime
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
        # true (mặc định, hành vi mới): vision CHỈ được dùng đúng 1 lần tại
        # cạnh lên của curve zone để làm điều kiện đầu; ngoài zone RouteEKF là
        # GPS+encoder thuần. false = hành vi cũ (anchor mỗi tick khi ngoài
        # zone) — chỉ để so sánh/hồi quy, KHÔNG khuyến nghị (xem docstring
        # đầu file: gán đè state 50 Hz, không gate, làm sigma_d giả).
        self.declare_parameter("anchor_on_curve_entry_only", True)
        self.declare_parameter("curve_zone_hysteresis_m", 2.0)
        self.declare_parameter("curve_zone_block_gps", True)
        # --- Curvature feed-forward (chỉ dùng trong curve zone, xem docstring) ---
        self.declare_parameter("curve_ff_preview_m", 1.5)
        self.declare_parameter("curve_zone_curvature_thresh", 0.05)
        # List [zone0, zone1, ...] theo thứ tự cua dọc tuyến (s tăng); 1 phần
        # tử = áp chung mọi zone. Xem RouteMapMatcher.detect_curve_zones.
        self.declare_parameter("curve_zone_dilate_before_m", [4.0])
        self.declare_parameter("curve_zone_dilate_after_m", [4.0])
        # --- Curvature profile (xem _curve_kappa_profile(), docstring đầu file) ---
        self.declare_parameter("kappa_profile_len_m", 12.0)
        self.declare_parameter("kappa_profile_step_m", 0.5)
        # --- Ghi CSV quỹ đạo localization (lat/lon) TRỰC TIẾP từ gps_node, khỏi
        # cần chạy script log riêng ở terminal 2. Cột: t_s, lat, lon,
        # in_curve_zone, s_m, d_m, sigma_d. Bật -> ghi mỗi log_latlon_rate_hz. ---
        self.declare_parameter("log_latlon_enable", False)
        self.declare_parameter("log_latlon_csv", "")  # rỗng = tự đặt ~/rl_car_run_<timestamp>.csv
        self.declare_parameter("log_latlon_rate_hz", 20.0)

        route_csv = self.get_parameter("route_csv").value
        if route_csv:
            csv_path = Path(route_csv)
        else:
            # Cài qua colcon thì node.py nằm trong install/, không phải source
            # tree -> phải lấy map/ qua share dir (đã đăng ký trong setup.py),
            # không suy từ __file__ như Gps.py (chạy trực tiếp từ source).
            from ament_index_python.packages import get_package_share_directory

            csv_path = Path(get_package_share_directory("RL_CAR")) / "map" / "gps_log.csv"
        self.matcher = RouteMapMatcher(csv_path)

        # Curve zones tính 1 lần từ hình học CSV lúc load (tuyến không đổi khi
        # đang chạy) — dùng để gate curvature feed-forward trong _tick(), xem
        # docstring đầu file. Không liên quan gate map-matching (update()).
        self.curve_ff_preview_m = float(self.get_parameter("curve_ff_preview_m").value)
        self.kappa_profile_len_m = float(self.get_parameter("kappa_profile_len_m").value)
        self.kappa_profile_step_m = float(self.get_parameter("kappa_profile_step_m").value)
        self._curve_zones = self.matcher.detect_curve_zones(
            curvature_thresh=float(self.get_parameter("curve_zone_curvature_thresh").value),
            dilate_before_m=list(self.get_parameter("curve_zone_dilate_before_m").value),
            dilate_after_m=list(self.get_parameter("curve_zone_dilate_after_m").value),
        )
        self.get_logger().info(
            f"Curve feed-forward: {len(self._curve_zones)} zone tren tuyen "
            f"({', '.join(f'{a:.0f}-{b:.0f}m' for a, b in self._curve_zones)})"
        )

        route_ekf_config = RouteEKFConfig(
            q_xy=float(self.get_parameter("q_xy").value),
            q_psi=float(self.get_parameter("q_psi").value),
            min_fix_type=int(self.get_parameter("min_fix_type").value),
            max_h_acc_m=float(self.get_parameter("max_h_acc_m").value),
            default_h_acc_m=float(self.get_parameter("default_h_acc_m").value),
            gps_gate_chi2=float(self.get_parameter("gps_gate_chi2").value),
            anchor_sigma_d=float(self.get_parameter("anchor_sigma_d").value),
            project_window_m=float(self.get_parameter("project_window_m").value),
            curve_zone_hysteresis_m=float(
                self.get_parameter("curve_zone_hysteresis_m").value
            ),
            curve_zone_block_gps=bool(
                self.get_parameter("curve_zone_block_gps").value
            ),
        )
        self.route_ekf = RouteEKF(self.matcher, route_ekf_config, curve_zones=self._curve_zones)
        # RouteEKF thứ 2 CHỈ predict() từ encoder, KHÔNG BAO GIỜ correct_gps()
        # — chạy song song để so sánh offline encoder trôi bao nhiêu nếu
        # không có GPS sửa (cột enc_* trong CSV log_latlon_enable, xem _tick()).
        # init cùng lúc, cùng điểm với route_ekf thật (xem _serial_loop).
        self.route_ekf_enc = RouteEKF(self.matcher, route_ekf_config, curve_zones=self._curve_zones)
        self._ekf_lock = threading.Lock()
        # Mẫu GPS THÔ mới nhất đọc được từ serial, TRƯỚC khi qua gate/correct
        # của route_ekf.correct_gps() — dùng để log so sánh (xem _tick()).
        self._last_raw_sample: tuple[float, float, int | None, float | None] | None = None

        self.vision_timeout_s = float(self.get_parameter("vision_timeout_s").value)
        self.anchor_on_curve_entry_only = bool(
            self.get_parameter("anchor_on_curve_entry_only").value
        )
        self._latest_odom: Odometry | None = None
        self._last_predict_time = self.get_clock().now()
        self._last_vision_fresh_time: float | None = None
        self._latest_camera_ekf_state: tuple[float, float, float, float] | None = None
        # Trạng thái curve-zone tick TRƯỚC — để phát hiện CẠNH LÊN (ngoài ->
        # trong zone), thời điểm DUY NHẤT vision được phép chạm vào state.
        self._prev_in_curve_zone = False

        best_effort_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.route_state_pub = self.create_publisher(
            Float64MultiArray, self.get_parameter("route_state_topic").value, 10
        )
        # Pose LOCALIZATION của hệ suy ngược ra lat/lon để lưu CSV, đánh giá
        # offline so với ground truth (RTK survey vạch thật). Trong cua đây là
        # encoder dead-reckon (GPS/vision chỉ neo điểm MỐC lúc vào cua rồi bị
        # chặn), tức đúng "toạ độ localization" mà hệ tin — xem
        # scripts/log_latlon.py (subscribe + ghi CSV song song full stack).
        # [lat, lon, in_curve_zone, s_m, d_m, sigma_d] — chỉ EKF fused, KHÔNG
        # có raw/encoder-thuần (2 cột đó chỉ ghi vào CSV qua log_latlon_enable,
        # xem writerow bên dưới + scripts/plot_route_compare.py).
        self.ekf_latlon_pub = self.create_publisher(Float64MultiArray, "/gps/ekf_latlon", 10)

        # Ghi CSV trực tiếp (nếu bật) — không cần script/terminal 2. Chỉ _tick
        # ghi (1 luồng writer duy nhất) nên không cần lock file.
        self._latlon_writer = None
        self._latlon_file = None
        self._latlon_t0: float | None = None
        self._latlon_last_write = 0.0
        self.log_latlon_rate_hz = max(1.0, float(self.get_parameter("log_latlon_rate_hz").value))
        if bool(self.get_parameter("log_latlon_enable").value):
            log_path = self.get_parameter("log_latlon_csv").value
            if not log_path:
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                log_path = str(Path.home() / f"rl_car_run_{stamp}.csv")
            try:
                self._latlon_file = open(log_path, "w", newline="")
                self._latlon_writer = csv.writer(self._latlon_file)
                self._latlon_writer.writerow(
                    [
                        "t_s", "lat", "lon", "in_curve_zone", "s_m", "d_m", "sigma_d",
                        # GPS THÔ (mẫu mới nhất từ serial, trước gate/correct_gps) —
                        # rỗng nếu chưa có fix nào từ lúc start log.
                        "raw_lat", "raw_lon", "raw_fix_type", "raw_h_acc_m",
                        # Encoder-thuần (route_ekf_enc, không bao giờ correct_gps) —
                        # so với EKF (s_m/d_m/sigma_d ở trên) để thấy GPS đóng góp
                        # bao nhiêu, đặc biệt xuyên qua curve zone.
                        "enc_s_m", "enc_d_m", "enc_sigma_d",
                        # lat/lon suy ngược từ (x,y) của route_ekf_enc — cho phép
                        # vẽ track dead-reckoning thuần LÊN BẢN ĐỒ (không chỉ đồ
                        # thị d/s), xem scripts/plot_run_map.py.
                        "enc_lat", "enc_lon",
                    ]
                )
                self.get_logger().info(f"Ghi quy dao lat/lon -> {log_path}")
            except OSError as exc:
                self.get_logger().error(f"Khong mo duoc file log {log_path}: {exc}")
                self._latlon_writer = None

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
                self._last_raw_sample = (
                    sample.lat, sample.lon, sample.fix_type, sample.h_acc_m,
                )
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
                        # Cùng điểm neo cho bản encoder-thuần -> 2 track khởi
                        # hành từ 1 chỗ, khác biệt sau đó chỉ đến từ GPS correct.
                        self.route_ekf_enc.initialize(s0, d0, psi_err=0.0)
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
                # Encoder-thuần: cùng predict, không bao giờ correct/anchor —
                # xem __init__ + docstring log_latlon_enable.
                self.route_ekf_enc.predict(
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

            # route_state() TRƯỚC: nó gọi _update_curve_zone_state() nên sau
            # dòng này in_curve_zone là trạng thái CỦA TICK NÀY (đã hysteresis),
            # không còn trễ 1 tick như bản cũ — cần độ chính xác đó để bắt đúng
            # CẠNH LÊN của zone.
            state = self.route_ekf.route_state()
            in_zone_now = self.route_ekf.in_curve_zone

            # Vision chỉ được chạm vào state ĐÚNG 1 LẦN, tại cạnh lên
            # (ngoài -> trong zone), làm điều kiện đầu cho đoạn cua. Ngoài
            # zone RouteEKF là GPS + encoder thuần; trong zone là encoder
            # thuần (GPS đã bị correct_gps() chặn bởi curve_zone_block_gps).
            # KHÔNG anchor khi đang Ở TRONG zone: lúc đó control_node đã chặn
            # vision correct FrenetEKF (curve mode tự lái), nên
            # /control/ekf_state chỉ còn là dead-reckon frame THẲNG đang trôi
            # theo độ cong — anchor bằng giá trị đó là nạp rác vào RouteEKF.
            entering_zone = in_zone_now and not self._prev_in_curve_zone
            allow_anchor = entering_zone if self.anchor_on_curve_entry_only else (not in_zone_now)
            if (
                allow_anchor
                and vision_fresh
                and self._latest_camera_ekf_state is not None
            ):
                _s, d_cam, psi_cam, _v = self._latest_camera_ekf_state
                self.route_ekf.anchor_lateral(d_m=d_cam, psi_err=psi_cam)
                # anchor_lateral() vừa ghi đè x/P -> đọc lại state cho khớp
                # với cái sẽ publish/log ngay bên dưới.
                state = self.route_ekf.route_state()
                if entering_zone:
                    self.get_logger().info(
                        f"Vao curve zone tai s={state.s_m:.1f}m: neo dieu kien dau tu vision "
                        f"d={d_cam:+.3f}m psi_err={math.degrees(psi_cam):+.1f}deg "
                        f"(tu day encoder dead-reckon, khong con vision/GPS)"
                    )
            elif entering_zone:
                # Vào cua mà vision không tươi -> không có gì để neo, giữ
                # nguyên d/psi hiện có (GPS+encoder đã dựng). Cảnh báo vì đây
                # là điều kiện đầu kém cho cả đoạn cua.
                self.get_logger().warn(
                    f"Vao curve zone tai s={state.s_m:.1f}m NHUNG vision khong tuoi "
                    f"-> khong neo duoc, dung d={state.d_m:+.3f}m hien co lam dieu kien dau"
                )
            self._prev_in_curve_zone = in_zone_now
            # Pose (x, y) local để suy ngược ra lat/lon — lấy trong lock cùng
            # state (to_wgs84 bên dưới là phép chiếu thuần, an toàn ngoài lock).
            ekf_x = float(self.route_ekf.x[0])
            ekf_y = float(self.route_ekf.x[1])
            enc_state = self.route_ekf_enc.route_state()
            enc_x = float(self.route_ekf_enc.x[0])
            enc_y = float(self.route_ekf_enc.x[1])
            raw_sample = self._last_raw_sample

        kappa_ff = self._curve_ff_kappa(state.s_m)
        # route_ekf.in_curve_zone: trạng thái HIỆN TẠI (không phải điểm
        # preview), đã áp hysteresis (xem RouteEKF._update_curve_zone_state,
        # gọi bên trong route_state() ở trên) — cùng 1 nguồn với cái
        # correct_gps() dùng để chặn GPS, không tính lại riêng bằng logic
        # khác dễ lệch nhau. Dùng cho control_node quyết định có ép chuyển
        # sang GPS+encoder ngay khi vào cua hay không (gps_force_in_curve_zone).
        # in_zone_now lấy TRONG lock cùng lúc với state (thay vì đọc lại
        # self.route_ekf.in_curve_zone ngoài lock) — đảm bảo cờ publish khớp
        # đúng state đang publish, không lệch nếu tick sau kịp chen vào.
        in_curve_zone = 1.0 if in_zone_now else 0.0
        kappa_profile = self._curve_kappa_profile(state.s_m)
        self.route_state_pub.publish(
            Float64MultiArray(
                data=[
                    state.s_m, state.d_m, state.psi_err, state.sigma_d,
                    kappa_ff, in_curve_zone, *kappa_profile,
                ]
            )
        )
        # Pose localization -> lat/lon (đánh giá offline vs ground truth RTK).
        lat, lon = self.matcher.to_wgs84(ekf_x, ekf_y)
        enc_lat, enc_lon = self.matcher.to_wgs84(enc_x, enc_y)
        self.ekf_latlon_pub.publish(
            Float64MultiArray(
                data=[lat, lon, in_curve_zone, state.s_m, state.d_m, state.sigma_d]
            )
        )
        # Ghi CSV trực tiếp (nếu bật) — throttle theo log_latlon_rate_hz.
        if self._latlon_writer is not None:
            now = time.monotonic()
            if (now - self._latlon_last_write) >= (1.0 / self.log_latlon_rate_hz):
                self._latlon_last_write = now
                if self._latlon_t0 is None:
                    self._latlon_t0 = now
                if raw_sample is None:
                    raw_lat = raw_lon = raw_fix = raw_hacc = ""
                else:
                    r_lat, r_lon, r_fix, r_hacc = raw_sample
                    raw_lat, raw_lon = f"{r_lat:.8f}", f"{r_lon:.8f}"
                    raw_fix = "" if r_fix is None else str(r_fix)
                    raw_hacc = "" if r_hacc is None else f"{r_hacc:.3f}"
                self._latlon_writer.writerow([
                    f"{now - self._latlon_t0:.3f}", f"{lat:.8f}", f"{lon:.8f}",
                    int(in_curve_zone), f"{state.s_m:.3f}",
                    f"{state.d_m:.4f}", f"{state.sigma_d:.4f}",
                    raw_lat, raw_lon, raw_fix, raw_hacc,
                    f"{enc_state.s_m:.3f}", f"{enc_state.d_m:.4f}", f"{enc_state.sigma_d:.4f}",
                    f"{enc_lat:.8f}", f"{enc_lon:.8f}",
                ])
                self._latlon_file.flush()

    def _curve_ff_kappa(self, s_m: float) -> float:
        """Độ cong feed-forward tại s_m + preview, quy ước panel (+ = phải).

        0.0 nếu điểm preview KHÔNG rơi vào curve zone nào — ngoài zone hình
        học tuyến gần thẳng, curvature_at() chỉ trả nhiễu ghi CSV, không nên
        cho nó vào tay lái (xem docstring đầu file)."""
        s_preview = s_m + self.curve_ff_preview_m
        if not any(a <= s_preview <= b for a, b in self._curve_zones):
            return 0.0
        return -self.matcher.curvature_at(s_preview)

    def _curve_kappa_profile(self, s_m: float) -> list[float]:
        """Profile độ cong feed-forward [step_m, n, kappa_0..kappa_n] dọc
        kappa_profile_len_m phía trước s_m, bước kappa_profile_step_m.

        Khác _curve_ff_kappa() (1 giá trị tại 1 điểm preview cố định), profile
        này cho control_node nội suy feed-forward tại ĐÚNG điểm pure pursuit
        đang nhắm (lookahead_distance thật, không phải curve_ff_preview_m) và
        bẻ cong ràng buộc max_curvature của planner theo đúng hình cua thật
        (xem planner_motion/frenet_planner.py: ref_kappa). Mỗi kappa_i gate
        theo curve zone y hệt _curve_ff_kappa() (0.0 ngoài zone, tránh nhiễu
        ghi CSV rung tay lái đoạn gần thẳng), cùng quy ước dấu panel (+=phải)."""
        step = self.kappa_profile_step_m
        n = max(0, int(round(self.kappa_profile_len_m / step)))
        profile: list[float] = [step, float(n)]
        for i in range(n + 1):
            s_i = s_m + i * step
            if any(a <= s_i <= b for a, b in self._curve_zones):
                profile.append(-self.matcher.curvature_at(s_i))
            else:
                profile.append(0.0)
        return profile

    def destroy_node(self) -> bool:
        self._stop_event.set()
        if self._reader_thread is not None:
            self._reader_thread.join(timeout=2.0)
        if self._serial is not None:
            try:
                self._serial.close()
            except Exception:
                pass
        if self._latlon_file is not None:
            try:
                self._latlon_file.close()
            except OSError:
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
