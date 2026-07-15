from __future__ import annotations

import json

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Float64, Float64MultiArray, String

from .logic import PerceptionConfig, PerceptionLogic
from .model_paths import default_seg_model_path, default_yolo_model_path


class PerceptionNode(Node):
    """Camera RGB+Depth -> seg làn -> Frenet -> detection vật cản -> chiếu vào Frenet.

    Không chạy planner (xem `planner_motion_node`) và không vẽ overlay chi tiết
    — chỉ blend sẵn segmentation lên RGB (visual_frame) và publish trạng thái
    Frenet + detections (frenet_state) để planner_motion_node tiêu thụ. Toàn bộ
    phần tính toán (seg/Frenet/detection) nằm trong `PerceptionLogic`
    (perception/logic.py) — node này chỉ lo subscribe/publish.
    """

    def __init__(self) -> None:
        super().__init__("perception_node")

        self.declare_parameter("rgb_topic", "/camera/camera/color/image_raw")
        self.declare_parameter("visual_frame_topic", "/perception/visual_frame")
        self.declare_parameter("frenet_state_topic", "/perception/frenet_state")
        self.declare_parameter("segmentation_alpha", 0.35)
        self.declare_parameter("yolo_model_path", default_yolo_model_path())
        self.declare_parameter("seg_model_path", default_seg_model_path())
        self.declare_parameter("device", "auto")
        self.declare_parameter("yolo_conf", 0.35)
        self.declare_parameter("yolo_iou", 0.45)
        self.declare_parameter("seg_conf", 0.35)
        self.declare_parameter("seg_x_min", 0.25)
        self.declare_parameter("seg_x_max", 0.75)
        self.declare_parameter("seg_min_area", 500)
        self.declare_parameter("seg_target_class_id", 1)  # 0: curb, 1: dashed_yellow_line
        self.declare_parameter("cam_cx", -1.0)    # -1 → auto (image_w/2)
        self.declare_parameter("seg_y_step", 4)  # sample spacing inside each dash (px)
        self.declare_parameter("frenet_alpha", 0.3)   # EMA weight for new measurement
        self.declare_parameter("lane_width_m", 2.7)  # total lane width in metres
        self.declare_parameter("depth_topic", "/camera/camera/aligned_depth_to_color/image_raw")
        self.declare_parameter(
            "camera_info_topic", "/camera/camera/color/camera_info"
        )
        self.declare_parameter("depth_scale", 0.001)  # metres per raw depth unit

        self.declare_parameter("frame_stride", 1)
        self.declare_parameter("obstacle_alpha", 0.05)
        self.declare_parameter("obstacle_track_ttl", 5)
        self.declare_parameter("obstacle_match_px", 90.0)
        self.declare_parameter("obstacle_match_s", 2.0)
        self.declare_parameter("obstacle_match_d", 1.2)
        self.declare_parameter("obstacle_match_cost_max", 1.35)
        self.declare_parameter("obstacle_depth_percentile", 35.0)
        self.declare_parameter("detection_conf_gate", 0.45)
        self.declare_parameter("seg_viz_all_classes", True)

        self.rgb_topic = self.get_parameter("rgb_topic").value
        self.visual_frame_topic = self.get_parameter("visual_frame_topic").value
        self.frenet_state_topic = self.get_parameter("frenet_state_topic").value
        self.depth_topic = self.get_parameter("depth_topic").value
        self.camera_info_topic = self.get_parameter("camera_info_topic").value
        self.frame_stride = max(1, int(self.get_parameter("frame_stride").value))

        self.bridge = CvBridge()
        self.frame_count = 0
        self._warned_missing_depth = False
        self._warned_missing_cam_info = False

        config = PerceptionConfig(
            yolo_model_path=self.get_parameter("yolo_model_path").value,
            seg_model_path=self.get_parameter("seg_model_path").value,
            device=str(self.get_parameter("device").value),
            yolo_conf=float(self.get_parameter("yolo_conf").value),
            yolo_iou=float(self.get_parameter("yolo_iou").value),
            seg_conf=float(self.get_parameter("seg_conf").value),
            seg_x_min=float(self.get_parameter("seg_x_min").value),
            seg_x_max=float(self.get_parameter("seg_x_max").value),
            seg_min_area=float(self.get_parameter("seg_min_area").value),
            seg_target_class_id=int(self.get_parameter("seg_target_class_id").value),
            cam_cx=float(self.get_parameter("cam_cx").value),
            seg_y_step=max(1, int(self.get_parameter("seg_y_step").value)),
            frenet_alpha=float(self.get_parameter("frenet_alpha").value),
            lane_width_m=float(self.get_parameter("lane_width_m").value),
            depth_scale=float(self.get_parameter("depth_scale").value),
            segmentation_alpha=float(self.get_parameter("segmentation_alpha").value),
            obstacle_alpha=float(self.get_parameter("obstacle_alpha").value),
            obstacle_track_ttl=max(1, int(self.get_parameter("obstacle_track_ttl").value)),
            obstacle_match_px=float(self.get_parameter("obstacle_match_px").value),
            obstacle_match_s=float(self.get_parameter("obstacle_match_s").value),
            obstacle_match_d=float(self.get_parameter("obstacle_match_d").value),
            obstacle_match_cost_max=float(self.get_parameter("obstacle_match_cost_max").value),
            obstacle_depth_percentile=float(self.get_parameter("obstacle_depth_percentile").value),
            detection_conf_gate=float(self.get_parameter("detection_conf_gate").value),
            seg_viz_all_classes=bool(self.get_parameter("seg_viz_all_classes").value),
        )
        self.logic = PerceptionLogic(config, logger=self.get_logger())

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        # RGB/depth ghi trong bag với QoS RELIABLE — nếu subscriber khai
        # BEST_EFFORT thì gói UDP rớt do hệ thống bận sẽ mất vĩnh viễn, không
        # log/error gì cả (đúng kiểu "rớt frame âm thầm" đang gặp). Dùng
        # RELIABLE cho input để DDS tự retransmit khi mất gói.
        input_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )
        # visual_frame là ảnh lớn (vd 1280x720 với camera thật) — dùng RELIABLE
        # giống input_qos để tránh rớt gói UDP âm thầm giữa 2 process
        # (perception_node -> visualization_node). frenet_state là JSON nhỏ,
        # giữ BEST_EFFORT cho nhẹ.
        self.visual_frame_pub = self.create_publisher(
            Image, self.visual_frame_topic, input_qos
        )
        self.frenet_state_pub = self.create_publisher(
            String, self.frenet_state_topic, qos
        )
        self.frenet_d_pub      = self.create_publisher(Float64,           "/perception/frenet/d",      10)
        self.frenet_hdg_pub    = self.create_publisher(Float64,           "/perception/frenet/heading", 10)
        self.frenet_coeffs_pub = self.create_publisher(Float64MultiArray, "/perception/frenet/coeffs",  10)

        # Mỗi subscription 1 callback_group riêng (MutuallyExclusive) để
        # image_callback (YOLO, có thể mất 20-100ms) không bao giờ chặn
        # _depth_callback/_cam_info_callback chạy trên MultiThreadedExecutor.
        image_group = MutuallyExclusiveCallbackGroup()
        depth_group = MutuallyExclusiveCallbackGroup()
        cam_info_group = MutuallyExclusiveCallbackGroup()
        self.image_sub = self.create_subscription(
            Image, self.rgb_topic, self.image_callback, input_qos, callback_group=image_group
        )
        self.create_subscription(
            Image,
            self.depth_topic,
            self._depth_callback,
            input_qos,
            callback_group=depth_group,
        )
        self.create_subscription(
            CameraInfo,
            self.camera_info_topic,
            self._cam_info_callback,
            10,
            callback_group=cam_info_group,
        )

        self.get_logger().info(
            "perception_node ready: rgb=%s depth=%s info=%s yolo=%s seg=%s device=%s"
            % (
                self.rgb_topic,
                self.depth_topic,
                self.camera_info_topic,
                config.yolo_model_path,
                config.seg_model_path,
                self.logic.device_name,
            )
        )

    def _depth_callback(self, msg: Image) -> None:
        raw = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        is_integer_encoding = msg.encoding == "16UC1" or np.issubdtype(raw.dtype, np.integer)
        self.logic.set_depth_image(raw, is_integer_encoding)

    def _cam_info_callback(self, msg: CameraInfo) -> None:
        k = list(msg.k)
        self.logic.set_camera_intrinsics(fx=k[0], fy=k[4], cx=k[2], cy=k[5])

    def image_callback(self, msg: Image) -> None:
        self.frame_count += 1
        if (self.frame_count - 1) % self.frame_stride != 0:
            return

        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:
            self.get_logger().error(f"Failed to convert RGB image: {exc}")
            return

        if not self.logic.has_depth:
            if not self._warned_missing_depth:
                self.get_logger().warning(
                    "Waiting for depth image before processing RGB frames."
                )
                self._warned_missing_depth = True
            return
        if not self.logic.has_camera_info:
            if not self._warned_missing_cam_info:
                self.get_logger().warning(
                    "Waiting for camera intrinsics before processing RGB frames."
                )
                self._warned_missing_cam_info = True
            return
        self._warned_missing_depth = False
        self._warned_missing_cam_info = False

        try:
            result = self.logic.process_frame(bgr, frame_index=self.frame_count)
            self._publish_outputs(msg, result)
        except Exception as exc:
            self.get_logger().error(f"Inference pipeline failed: {exc}")

    def _publish_outputs(self, input_msg: Image, result) -> None:
        frame_msg = self.bridge.cv2_to_imgmsg(result.visual_frame, encoding="bgr8")
        frame_msg.header = input_msg.header
        self.visual_frame_pub.publish(frame_msg)

        payload = {
            "frame_id": input_msg.header.frame_id,
            "stamp": {
                "sec": int(input_msg.header.stamp.sec),
                "nanosec": int(input_msg.header.stamp.nanosec),
            },
            "segmentation": result.segmentation_meta,
            "detections": result.detections,
            "frenet_viz": result.frenet_viz,
        }
        self.frenet_state_pub.publish(String(data=json.dumps(payload)))

        if result.frenet_viz is None:
            return
        self.frenet_d_pub.publish(Float64(data=result.d_meters_filtered))
        self.frenet_hdg_pub.publish(Float64(data=result.heading_filtered))
        coeffs_msg = Float64MultiArray()
        coeffs_msg.data = result.coeffs
        self.frenet_coeffs_pub.publish(coeffs_msg)


def main(args: list[str] | None = None) -> None:
    cv2.setNumThreads(1)
    rclpy.init(args=args)
    node = PerceptionNode()
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
