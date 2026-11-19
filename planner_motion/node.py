from __future__ import annotations

import json

import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Float64, Float64MultiArray, String

from .logic import PlannerLogic


class PlannerMotionNode(Node):
    """Frenet Optimal Planner tách riêng khỏi perception_node.

    Planner sinh hàng trăm quỹ đạo ứng viên mỗi frame (polynomial bậc 5, thuần
    Python/numpy) — tốn CPU đáng kể. Tách ra process riêng để không chặn
    pipeline YOLO của perception_node và để 2 việc chạy song song trên các
    core khác nhau. Thuật toán nằm trong `PlannerLogic` (planner_motion/logic.py)
    — node này chỉ lo subscribe/publish.

    Subscribe `frenet_state_topic` (trạng thái Frenet đã lọc + detections kèm
    obstacle từ perception_node), chạy planner, rồi publish lại đúng payload
    đó nhưng đã gắn thêm optimal_path/candidate_paths lên `visual_frenet_topic`
    — visualization_node không cần đổi gì, vẫn nhận đúng topic như trước.
    """

    def __init__(self) -> None:
        super().__init__("planner_motion_node")

        self.declare_parameter("frenet_state_topic", "/perception/frenet_state")
        self.declare_parameter("visual_frenet_topic", "/perception/visual_frenet")
        self.declare_parameter("plan_enable", True)
        self.declare_parameter("plan_speed", 2.0)          # vận tốc giả định/đích [m/s]
        self.declare_parameter("plan_robot_radius", 0.6)   # bán kính an toàn [m]
        self.declare_parameter("plan_road_width", 2.5)     # nửa bề rộng lấy mẫu ngang [m]
        self.declare_parameter("plan_max_curvature", 1.5)  # [1/m]
        self.declare_parameter("plan_lookahead", 1.5)      # s để lấy target_d [m]
        self.declare_parameter("plan_clearance", 1.2)      # khoảng cách mềm né vật [m]
        self.declare_parameter("plan_obstacle_weight", 10.0)  # độ mạnh phạt né vật
        self.declare_parameter("plan_center_offset", 0.0)  # +d = lệch điểm giữa sang phải [m]
        self.declare_parameter("plan_center_weight", 1.0)  # trọng số ưu tiên bám center_offset (k_d)
        self.declare_parameter("plan_min_horizon_s", 3.5)  # horizon tối thiểu polynomial ngang [s]
        self.declare_parameter("plan_max_horizon_s", 4.0)  # horizon tối đa polynomial ngang [s]
        self.declare_parameter("plan_d_road_w", 0.4)        # bước lấy mẫu di (lệch ngang ứng viên) [m]

        self.frenet_state_topic = self.get_parameter("frenet_state_topic").value
        self.visual_frenet_topic = self.get_parameter("visual_frenet_topic").value

        self.logic = PlannerLogic(
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

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.visual_frenet_pub = self.create_publisher(
            String, self.visual_frenet_topic, qos
        )
        self.opt_path_pub = self.create_publisher(
            Float64MultiArray, "/perception/frenet/optimal_path", 10
        )
        self.opt_target_d_pub = self.create_publisher(
            Float64, "/perception/frenet/target_d", 10
        )
        self.create_subscription(
            String, self.frenet_state_topic, self._frenet_state_callback, qos
        )

        self.get_logger().info(
            "planner_motion_node ready: frenet_state=%s visual_frenet=%s plan_enable=%s"
            % (self.frenet_state_topic, self.visual_frenet_topic, self.logic.plan_enable)
        )

    def _frenet_state_callback(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError as exc:
            self.get_logger().error(f"Failed to parse frenet_state payload: {exc}")
            return

        frenet = payload.get("frenet_viz")
        detections = payload.get("detections", [])
        best_path = self.logic.plan(frenet, detections) if frenet is not None else None
        self._publish_optimal_path(best_path)
        self.visual_frenet_pub.publish(String(data=json.dumps(payload)))

    def _publish_optimal_path(self, best_path) -> None:
        msg = Float64MultiArray()
        msg.data = self.logic.best_path_to_flat_array(best_path)
        self.opt_path_pub.publish(msg)
        target_d = self.logic.best_path_target_d(best_path)
        if target_d is not None:
            self.opt_target_d_pub.publish(Float64(data=target_d))


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = PlannerMotionNode()
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
