"""ROS2 node sở hữu cổng serial hoverboard: đọc feedback -> publish /odom,
nhận /cmd_vel -> gửi lệnh steer/speed xuống xe.

Cổng serial chỉ mở được bởi 1 process nên node này vừa đọc encoder vừa
gửi lệnh điều khiển (control_node chỉ tính toán và publish /cmd_vel, không
đụng tới serial).
"""
from __future__ import annotations

import math

import rclpy
import serial
from geometry_msgs.msg import Quaternion, TransformStamped, Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from tf2_ros import TransformBroadcaster

from .logic import CmdVelToRawConfig, DifferentialOdometry, OdomConfig, cmd_vel_to_raw
from .protocol import decode_feedback_frame, parse_feedback, send_command


def yaw_to_quaternion(yaw: float) -> Quaternion:
    q = Quaternion()
    q.z = math.sin(yaw / 2.0)
    q.w = math.cos(yaw / 2.0)
    return q


class EncoderNode(Node):
    def __init__(self) -> None:
        super().__init__("encoder_node")

        self.declare_parameter("serial_port", "/dev/ttyUSB0")
        self.declare_parameter("baud_rate", 115200)
        self.declare_parameter("wheel_radius", 0.0762)
        self.declare_parameter("track_width", 0.3556)
        self.declare_parameter("right_wheel_sign", -1.0)
        self.declare_parameter("poll_rate_hz", 150.0)
        self.declare_parameter("odom_publish_rate_hz", 50.0)
        self.declare_parameter("cmd_vel_timeout", 0.3)
        self.declare_parameter("frame_id", "odom")
        self.declare_parameter("child_frame_id", "base_link")
        self.declare_parameter("publish_tf", True)
        # Khớp tỉ lệ với Encoder/wireless.py: full cần ga (max_linear_speed
        # m/s) -> speed_raw=50, full cần lái (max_angular_speed rad/s) ->
        # steer_raw=40 (= MAX_SPEED/MAX_STEER bên wireless.py).
        self.declare_parameter("cmd_speed_scale", 50.0)
        self.declare_parameter("cmd_steer_scale", 26.7)
        self.declare_parameter("max_speed_raw", 50)
        self.declare_parameter("max_steer_raw", 40)

        self.frame_id = self.get_parameter("frame_id").value
        self.child_frame_id = self.get_parameter("child_frame_id").value
        self.publish_tf = bool(self.get_parameter("publish_tf").value)
        self.cmd_vel_timeout = float(self.get_parameter("cmd_vel_timeout").value)

        self.odom = DifferentialOdometry(OdomConfig(
            wheel_radius=float(self.get_parameter("wheel_radius").value),
            track_width=float(self.get_parameter("track_width").value),
            right_wheel_sign=float(self.get_parameter("right_wheel_sign").value),
        ))
        self.cmd_cfg = CmdVelToRawConfig(
            cmd_speed_scale=float(self.get_parameter("cmd_speed_scale").value),
            cmd_steer_scale=float(self.get_parameter("cmd_steer_scale").value),
            max_speed_raw=int(self.get_parameter("max_speed_raw").value),
            max_steer_raw=int(self.get_parameter("max_steer_raw").value),
        )

        port = self.get_parameter("serial_port").value
        baud = int(self.get_parameter("baud_rate").value)
        try:
            self._ser = serial.Serial(port, baud, timeout=0.0)
            self.get_logger().info(f"Đã mở cổng serial {port} @ {baud}")
        except Exception as exc:
            self.get_logger().error(f"Không mở được cổng serial {port}: {exc}")
            self._ser = None

        self._rx_buffer = b""
        self._last_odom_time = self.get_clock().now()
        self._last_cmd = (0.0, 0.0)
        self._last_cmd_time = None

        self.odom_pub = self.create_publisher(Odometry, "/odom", 10)
        self.tf_broadcaster = TransformBroadcaster(self) if self.publish_tf else None
        self.create_subscription(Twist, "/cmd_vel", self._cmd_vel_cb, 10)

        poll_rate = float(self.get_parameter("poll_rate_hz").value)
        odom_rate = float(self.get_parameter("odom_publish_rate_hz").value)
        self.create_timer(1.0 / poll_rate, self._poll_serial)
        self.create_timer(0.05, self._send_drive_command)  # 20Hz, khớp TIME_SEND firmware
        self.create_timer(1.0 / odom_rate, self._publish_odom)

        self.get_logger().info("encoder_node ready")

    def _cmd_vel_cb(self, msg: Twist) -> None:
        self._last_cmd = (msg.linear.x, msg.angular.z)
        self._last_cmd_time = self.get_clock().now()

    def _poll_serial(self) -> None:
        if self._ser is None:
            return
        try:
            if self._ser.in_waiting > 0:
                self._rx_buffer += self._ser.read(self._ser.in_waiting)
                self._rx_buffer = parse_feedback(self._rx_buffer, self._on_feedback_frame)
        except Exception as exc:
            self.get_logger().error(f"Lỗi đọc serial: {exc}")

    def _on_feedback_frame(self, decoded: dict) -> None:
        now = self.get_clock().now()
        dt = (now - self._last_odom_time).nanoseconds * 1e-9
        self._last_odom_time = now
        self.odom.update(decoded["speedL_rpm"], decoded["speedR_rpm"], dt)

    def _send_drive_command(self) -> None:
        if self._ser is None:
            return
        now = self.get_clock().now()
        stale = (self._last_cmd_time is None or
                  (now - self._last_cmd_time).nanoseconds * 1e-9 > self.cmd_vel_timeout)
        linear_x, angular_z = (0.0, 0.0) if stale else self._last_cmd
        steer_raw, speed_raw = cmd_vel_to_raw(linear_x, angular_z, self.cmd_cfg)
        try:
            send_command(self._ser, steer_raw, speed_raw)
        except Exception as exc:
            self.get_logger().error(f"Lỗi gửi lệnh serial: {exc}")

    def _publish_odom(self) -> None:
        stamp = self.get_clock().now().to_msg()
        x, y, theta, v, omega = self.odom.x, self.odom.y, self.odom.theta, self.odom.v, self.odom.omega

        msg = Odometry()
        msg.header.stamp = stamp
        msg.header.frame_id = self.frame_id
        msg.child_frame_id = self.child_frame_id
        msg.pose.pose.position.x = x
        msg.pose.pose.position.y = y
        msg.pose.pose.orientation = yaw_to_quaternion(theta)
        msg.twist.twist.linear.x = v
        msg.twist.twist.angular.z = omega
        self.odom_pub.publish(msg)

        if self.tf_broadcaster is not None:
            t = TransformStamped()
            t.header.stamp = stamp
            t.header.frame_id = self.frame_id
            t.child_frame_id = self.child_frame_id
            t.transform.translation.x = x
            t.transform.translation.y = y
            t.transform.rotation = yaw_to_quaternion(theta)
            self.tf_broadcaster.sendTransform(t)

    def destroy_node(self) -> bool:
        if self._ser is not None:
            try:
                send_command(self._ser, 0, 0)
                self._ser.close()
            except Exception:
                pass
        return super().destroy_node()


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = EncoderNode()
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
