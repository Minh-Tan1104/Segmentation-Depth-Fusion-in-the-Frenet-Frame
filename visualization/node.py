from __future__ import annotations

import json
from collections import deque
from typing import Any

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray, String

from Gps.map_matcher import RouteMapMatcher

from .logic import OverlayRenderer


class VisualizationNode(Node):
    """Vẽ overlay (làn đường, vật cản, panel Frenet, quỹ đạo tối ưu).

    Không chạy model — chỉ tiêu thụ 2 topic do perception_node publish ra:
    `visual_frame` (ảnh RGB đã blend sẵn segmentation) và `visual_frenet`
    (JSON detections + frenet_viz). Render ngay khi có `visual_frame` mới,
    dùng dữ liệu `visual_frenet` mới nhất đã cache — không khớp stamp tuyệt
    đối, để không bao giờ rớt khung hình chỉ vì 2 topic lệch nhau vài ms. Toàn
    bộ phần vẽ nằm trong `OverlayRenderer` (visualization/logic.py).
    """

    def __init__(self) -> None:
        super().__init__("visualization_node")

        self.declare_parameter("visual_frame_topic", "/perception/visual_frame")
        self.declare_parameter("visual_frenet_topic", "/perception/visual_frenet")
        self.declare_parameter("overlay_topic", "/perception/overlay")
        self.declare_parameter("lane_width_m", 2.7)
        self.declare_parameter("use_window", False)
        self.declare_parameter("window_name", "RGB Dual Model View")
        self.declare_parameter("window_scale", 0.6)
        self.declare_parameter("odom_topic", "/odom")
        self.declare_parameter("ekf_state_topic", "/control/ekf_state")
        self.declare_parameter("planner_viz_topic", "/control/planner_viz")
        self.declare_parameter("encoder_trail_max_points", 1500)
        self.declare_parameter("gps_route_state_topic", "/gps/route_state")
        self.declare_parameter("route_csv", "")  # rỗng = map/gps_log.csv mặc định
        # Cùng key với gps_node (khai báo ở khối /** trong yaml) để zone vẽ ra
        # khớp đúng đoạn cua mà gps_node dùng cho curvature feed-forward.
        self.declare_parameter("curve_zone_curvature_thresh", 0.05)
        # List [zone0, zone1, ...] theo thứ tự cua dọc tuyến; 1 phần tử = áp
        # chung. Xem RouteMapMatcher.detect_curve_zones — PHẢI khớp gps_node
        # để đoạn cua tô màu trên panel khớp đúng đoạn feed-forward thật.
        self.declare_parameter("curve_zone_dilate_before_m", [4.0])
        self.declare_parameter("curve_zone_dilate_after_m", [4.0])
        # Panel Frenet khi control_node đang ở curve mode (curve_frame trong
        # planner_viz): mặc định true = khung nhìn ĐÚNG đoạn cua đang chạy
        # (waypoint đầu -> waypoint cuối của curve zone chứa xe, không phải
        # toàn tuyến 551m) — xem draw_curve_map_panel. false = toàn tuyến.
        self.declare_parameter("curve_panel_zoom_zone", True)
        self.declare_parameter("curve_panel_zone_margin_m", 2.0)
        self.declare_parameter("display_width", 1280)
        self.declare_parameter("display_height", 720)

        self.visual_frame_topic = self.get_parameter("visual_frame_topic").value
        self.visual_frenet_topic = self.get_parameter("visual_frenet_topic").value
        self.overlay_topic = self.get_parameter("overlay_topic").value
        self.lane_width_m = float(self.get_parameter("lane_width_m").value)
        self.use_window = bool(self.get_parameter("use_window").value)
        self.window_name = self.get_parameter("window_name").value
        self.window_scale = max(0.1, float(self.get_parameter("window_scale").value))
        self.odom_topic = self.get_parameter("odom_topic").value
        self.ekf_state_topic = self.get_parameter("ekf_state_topic").value
        self.planner_viz_topic = self.get_parameter("planner_viz_topic").value
        self.gps_route_state_topic = self.get_parameter("gps_route_state_topic").value
        self.curve_panel_zoom_zone = bool(self.get_parameter("curve_panel_zoom_zone").value)
        self.curve_panel_zone_margin_m = float(
            self.get_parameter("curve_panel_zone_margin_m").value
        )
        self.display_width = max(320, int(self.get_parameter("display_width").value))
        self.display_height = max(240, int(self.get_parameter("display_height").value))

        # Chỉ cần polyline để vẽ map panel — không cần EKF/serial gì ở đây,
        # instance riêng của node này, độc lập với Gps/node.py.
        route_csv = self.get_parameter("route_csv").value
        if route_csv:
            from pathlib import Path
            route_csv_path = Path(route_csv)
        else:
            from ament_index_python.packages import get_package_share_directory
            from pathlib import Path
            route_csv_path = Path(get_package_share_directory("RL_CAR")) / "map" / "gps_log.csv"
        try:
            self._gps_matcher: RouteMapMatcher | None = RouteMapMatcher(route_csv_path)
        except Exception as exc:
            self.get_logger().warn(f"Khong nap duoc route CSV cho GPS panel: {exc}")
            self._gps_matcher = None
        self._gps_curve_zones: list[tuple[float, float]] = (
            self._gps_matcher.detect_curve_zones(
                curvature_thresh=float(self.get_parameter("curve_zone_curvature_thresh").value),
                dilate_before_m=list(self.get_parameter("curve_zone_dilate_before_m").value),
                dilate_after_m=list(self.get_parameter("curve_zone_dilate_after_m").value),
            )
            if self._gps_matcher is not None
            else []
        )
        self._latest_gps_route_state: list[float] | None = None

        self.bridge = CvBridge()
        self.renderer = OverlayRenderer(self.lane_width_m)
        self._window_initialized = False
        self._latest_payload: dict[str, Any] | None = None
        # Trail (x, y) từ /odom — gốc tọa độ là điểm xuất phát (encoder_node
        # luôn khởi tạo x=y=0), dùng để xác nhận encoder + EKF còn hoạt động
        # khi mất line (vẽ ở draw_encoder_panel, visualization/logic.py).
        trail_max = max(2, int(self.get_parameter("encoder_trail_max_points").value))
        self._odom_trail: deque[tuple[float, float]] = deque(maxlen=trail_max)
        self._latest_ekf_state: list[float] | None = None
        # Path do planner NỘI BỘ của control_node sinh (EKF-driven, pipeline
        # thật đang lái xe) — khác với `_latest_payload["frenet_viz"]` (từ
        # planner_motion_node, chỉ dựa đo trực tiếp camera, dùng cho viz/rosbag).
        self._latest_planner_viz: dict[str, Any] | None = None

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        # visual_frame là ảnh lớn, perception_node publish bằng RELIABLE —
        # subscriber phải khớp RELIABLE, không thì rớt gói UDP âm thầm giữa
        # 2 process (đúng nguyên nhân gây rớt frame ở overlay).
        frame_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )
        self.overlay_pub = self.create_publisher(Image, self.overlay_topic, 1)
        self.create_subscription(
            Image, self.visual_frame_topic, self._frame_callback, frame_qos
        )
        self.create_subscription(
            String, self.visual_frenet_topic, self._frenet_callback, qos
        )
        self.create_subscription(Odometry, self.odom_topic, self._odom_callback, 10)
        self.create_subscription(
            Float64MultiArray, self.ekf_state_topic, self._ekf_state_callback, 10
        )
        self.create_subscription(
            String, self.planner_viz_topic, self._planner_viz_callback, 10
        )
        self.create_subscription(
            Float64MultiArray,
            self.gps_route_state_topic,
            self._gps_route_state_callback,
            10,
        )

        self.get_logger().info(
            "visualization_node ready: visual_frame=%s visual_frenet=%s overlay=%s"
            % (self.visual_frame_topic, self.visual_frenet_topic, self.overlay_topic)
        )

    def _frame_callback(self, msg: Image) -> None:
        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:
            self.get_logger().error(f"Failed to convert visual_frame image: {exc}")
            return

        payload = self._latest_payload or {}
        detections = payload.get("detections", [])
        frenet = payload.get("frenet_viz")
        if frenet is not None:
            frenet.setdefault("lane_width_m", self.lane_width_m)

        try:
            overlay = self.renderer.render(bgr, detections, frenet)
        except Exception as exc:
            self.get_logger().error(f"Overlay rendering failed: {exc}")
            return

        overlay_msg = self.bridge.cv2_to_imgmsg(overlay, encoding="bgr8")
        overlay_msg.header = msg.header
        self.overlay_pub.publish(overlay_msg)

        if self.use_window:
            self._show_window(overlay, frenet, detections)

    def _frenet_callback(self, msg: String) -> None:
        try:
            self._latest_payload = json.loads(msg.data)
        except json.JSONDecodeError as exc:
            self.get_logger().error(f"Failed to parse visual_frenet payload: {exc}")

    def _odom_callback(self, msg: Odometry) -> None:
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        self._odom_trail.append((x, y))

    def _ekf_state_callback(self, msg: Float64MultiArray) -> None:
        if len(msg.data) >= 2:
            self._latest_ekf_state = list(msg.data)

    def _gps_route_state_callback(self, msg: Float64MultiArray) -> None:
        if len(msg.data) >= 4:
            self._latest_gps_route_state = list(msg.data)  # [s_m, d_m, psi_err, sigma_d]

    def _planner_viz_callback(self, msg: String) -> None:
        try:
            self._latest_planner_viz = json.loads(msg.data)
        except json.JSONDecodeError as exc:
            self.get_logger().error(f"Failed to parse planner_viz payload: {exc}")

    def _ensure_window(self) -> None:
        if self._window_initialized:
            return
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        self._window_initialized = True

    def _show_window(
        self,
        overlay: np.ndarray,
        frenet: dict[str, Any] | None,
        detections: list[dict[str, Any]],
    ) -> None:
        """Lưới 2x2 vừa đúng display_width x display_height (mặc định
        1280x720, mỗi ô 640x360): [Camera | Planner/EKF] / [Encoder | GPS].
        Bỏ panel "Frenet (perception)" riêng — planner_motion_node chỉ phục
        vụ rosbag/so sánh, không phải pipeline thật đang lái xe (xem
        control/node.py _planner_tick); panel Planner/EKF ở trên đã đủ."""
        self._ensure_window()
        pad = 10
        cell_w = self.display_width // 2
        cell_h = self.display_height // 2

        cam_panel = cv2.resize(overlay, (cell_w, cell_h), interpolation=cv2.INTER_AREA)

        # Planner NỘI BỘ control_node — EKF-driven, đúng path/pose thật đang
        # dùng để tính cmd_vel (xem control/node.py _planner_tick). Khi
        # control_node ở curve mode (curve_frame=true trong payload): chuyển
        # sang panel CSV-frame — vẽ tuyến từ điểm đầu tới điểm cuối CSV, xe/
        # path đặt đúng toạ độ map, KHÔNG còn ego-frame (yêu cầu thiết kế
        # curve mode, xem control/node.py:_planner_tick_curve).
        control_panel = np.full((cell_h, cell_w, 3), 18, dtype=np.uint8)
        pv = self._latest_planner_viz
        # detections: perception_node đo 1 lần, planner_motion_node chỉ pass-
        # through nguyên vẹn qua /perception/visual_frenet (self._latest_payload)
        # — cùng dữ liệu vật cản thật bất kể control_node đang chạy planner nào,
        # nên dùng lại được cho panel control (trước đây truyền [] rỗng, chấm
        # đỏ vật cản không bao giờ hiện ở panel này dù draw_frenet_panel đã vẽ
        # sẵn logic đó).
        detections = (self._latest_payload or {}).get("detections", [])
        if (
            pv is not None
            and pv.get("curve_frame")
            and self._gps_matcher is not None
        ):
            self.renderer.draw_curve_map_panel(
                control_panel, self._gps_matcher.route_xy_local, pv,
                self._gps_curve_zones, pad, pad,
                cell_w - 2 * pad, cell_h - 2 * pad,
                zoom_zone=self.curve_panel_zoom_zone,
                zone_margin_m=self.curve_panel_zone_margin_m,
            )
        else:
            self.renderer.draw_frenet_panel(
                control_panel, pv, detections, pad, pad,
                cell_w - 2 * pad, cell_h - 2 * pad, title="Frenet/EKF (control)",
            )

        encoder_panel = np.full((cell_h, cell_w, 3), 18, dtype=np.uint8)
        self.renderer.draw_encoder_panel(
            encoder_panel, list(self._odom_trail), self._latest_ekf_state,
            pad, pad, cell_w - 2 * pad, cell_h - 2 * pad,
        )

        gps_panel = np.full((cell_h, cell_w, 3), 18, dtype=np.uint8)
        car_xy = car_heading = sigma_d = None
        if self._gps_matcher is not None and self._latest_gps_route_state is not None:
            s_m, d_m, psi_err, sigma_d = self._latest_gps_route_state[:4]
            rx, ry = self._gps_matcher.point_at(s_m)
            theta = self._gps_matcher.heading_at(s_m)
            car_xy = (rx + d_m * np.sin(theta), ry - d_m * np.cos(theta))
            car_heading = theta - psi_err
        route_xy = self._gps_matcher.route_xy_local if self._gps_matcher is not None else None
        self.renderer.draw_gps_panel(
            gps_panel, route_xy, car_xy, car_heading, sigma_d,
            pad, pad, cell_w - 2 * pad, cell_h - 2 * pad,
            curve_zones=self._gps_curve_zones,
        )

        top = np.hstack((cam_panel, control_panel))
        bottom = np.hstack((encoder_panel, gps_panel))
        frame = np.vstack((top, bottom))
        if abs(self.window_scale - 1.0) > 1e-6:
            target_w = max(1, int(round(frame.shape[1] * self.window_scale)))
            target_h = max(1, int(round(frame.shape[0] * self.window_scale)))
            interp = cv2.INTER_AREA if self.window_scale < 1.0 else cv2.INTER_LINEAR
            frame = cv2.resize(frame, (target_w, target_h), interpolation=interp)
        cv2.imshow(self.window_name, frame)
        cv2.waitKey(1)


def main(args: list[str] | None = None) -> None:
    cv2.setNumThreads(1)
    rclpy.init(args=args)
    node = VisualizationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node.use_window:
            cv2.destroyAllWindows()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
