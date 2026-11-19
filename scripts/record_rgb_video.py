#!/usr/bin/env python3
"""Record RGB video from a running RealSense camera (ROS 2 topic) to a file.

Tuỳ chọn --drive bật luôn phần lái xe bằng tay cầm PS4 qua serial (logic lấy
từ Encoder/wireless.py) ngay trong cùng tiến trình, để vừa lái vừa ghi hình mà
không cần chạy thêm control_node/encoder_node ROS.

Usage:
    # Terminal 1: start the camera
    ros2 launch RL_CAR realsense.launch.py

    # Terminal 2: chỉ ghi hình
    python3 scripts/record_rgb_video.py -o output.mp4 --fps 30

    # Terminal 2: vừa lái (tay cầm PS4 qua serial) vừa ghi hình
    python3 scripts/record_rgb_video.py -o output.mp4 --drive --port /dev/ttyUSB0
"""

from __future__ import annotations

import argparse
import struct
import time
from pathlib import Path
import sys

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image

try:
    import pygame
except ImportError:
    pygame = None

try:
    import serial
except ImportError:
    serial = None

DEFAULT_TOPIC = "/camera/camera/color/image_raw"

# ================== CẤU HÌNH LÁI (giống Encoder/wireless.py) ==================
START_FRAME = 0xABCD
MAX_SPEED = 50
MAX_STEER = 50
DEADZONE = 0.15
AXIS_SPEED = 1   # Left stick Y -> tiến/lùi
AXIS_STEER = 2   # Right stick X -> rẽ trái/phải (yaw)
BUTTON_EXIT = 9  # Nút "Options" -> thoát


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Subscribe to a RealSense RGB topic and record it to a video file."
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("rgb_recording.mp4"),
        help="Output video path. Default: rgb_recording.mp4",
    )
    parser.add_argument(
        "-t",
        "--topic",
        default=DEFAULT_TOPIC,
        help=f"RGB image topic to subscribe to. Default: {DEFAULT_TOPIC}",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Output video frame rate. Default: 30",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="Stop recording automatically after N seconds. Default: 0 means run until Ctrl+C.",
    )
    parser.add_argument(
        "--drive",
        action="store_true",
        help="Đồng thời lái xe bằng tay cầm PS4 qua serial (giống Encoder/wireless.py).",
    )
    parser.add_argument(
        "--port",
        default="/dev/ttyUSB0",
        help="Cổng serial Hoverboard khi dùng --drive. Default: /dev/ttyUSB0",
    )
    parser.add_argument(
        "--baud",
        type=int,
        default=115200,
        help="Baud rate serial khi dùng --drive. Default: 115200",
    )
    return parser.parse_args()


def send_drive_command(ser, steer: int, speed: int) -> None:
    checksum = (START_FRAME ^ (steer & 0xFFFF) ^ (speed & 0xFFFF)) & 0xFFFF
    packet = struct.pack("<HhhH", START_FRAME, steer, speed, checksum)
    ser.write(packet)


def apply_deadzone(value: float, deadzone: float = DEADZONE) -> float:
    if abs(value) < deadzone:
        return 0.0
    sign = 1.0 if value > 0 else -1.0
    return sign * (abs(value) - deadzone) / (1.0 - deadzone)


class RgbVideoRecorder(Node):
    def __init__(
        self,
        topic: str,
        output_path: Path,
        fps: float,
        duration: float,
        drive: bool = False,
        serial_port: str = "/dev/ttyUSB0",
        baud_rate: int = 115200,
    ) -> None:
        super().__init__("rgb_video_recorder")
        self.bridge = CvBridge()
        self.output_path = output_path
        self.fps = fps
        self.duration = duration
        self.writer: cv2.VideoWriter | None = None
        self.frame_count = 0
        self.start_time = self.get_clock().now()

        qos = QoSProfile(
            depth=5,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
        )
        self.subscription = self.create_subscription(Image, topic, self._on_image, qos)
        self.get_logger().info(f"Subscribing to {topic}, writing to {output_path}")

        self._ser = None
        self._js = None
        self._last_send_time = 0.0
        if drive:
            self._setup_drive(serial_port, baud_rate)

    def _setup_drive(self, serial_port: str, baud_rate: int) -> None:
        if serial is None:
            raise SystemExit("--drive cần package 'pyserial' (pip install pyserial).")
        if pygame is None:
            raise SystemExit("--drive cần package 'pygame' (pip install pygame).")

        self._ser = serial.Serial(serial_port, baud_rate, timeout=0.01)
        self.get_logger().info(f"Đã kết nối serial Hoverboard: {serial_port}@{baud_rate}")

        pygame.init()
        pygame.joystick.init()
        if pygame.joystick.get_count() == 0:
            raise SystemExit("Không tìm thấy tay cầm PS4. Cắm USB/ghép Bluetooth rồi thử lại.")
        self._js = pygame.joystick.Joystick(0)
        self._js.init()
        self.get_logger().info(f"Đã nhận diện tay cầm: {self._js.get_name()}")

        self.create_timer(0.02, self._on_drive_tick)

    def _on_drive_tick(self) -> None:
        pygame.event.pump()

        if self._js.get_numbuttons() > BUTTON_EXIT and self._js.get_button(BUTTON_EXIT):
            self.get_logger().info("Đã nhấn nút thoát trên tay cầm...")
            raise KeyboardInterrupt

        raw_speed_axis = -self._js.get_axis(AXIS_SPEED)
        raw_steer_axis = self._js.get_axis(AXIS_STEER)

        speed = int(apply_deadzone(raw_speed_axis) * MAX_SPEED)
        steer = int(apply_deadzone(raw_steer_axis) * MAX_STEER)

        now = time.time()
        if now - self._last_send_time >= 0.05:
            send_drive_command(self._ser, steer, speed)
            self._last_send_time = now

    def _on_image(self, msg: Image) -> None:
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

        if self.writer is None:
            height, width = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.writer = cv2.VideoWriter(str(self.output_path), fourcc, self.fps, (width, height))
            if not self.writer.isOpened():
                raise SystemExit(f"Failed to open video writer for: {self.output_path}")

        self.writer.write(frame)
        self.frame_count += 1

        if self.duration > 0:
            elapsed = (self.get_clock().now() - self.start_time).nanoseconds / 1e9
            if elapsed >= self.duration:
                raise KeyboardInterrupt

    def close(self) -> None:
        if self.writer is not None:
            self.writer.release()
        self.get_logger().info(f"Saved {self.frame_count} frames to {self.output_path}")
        if self._ser is not None:
            send_drive_command(self._ser, 0, 0)  # dừng xe trước khi thoát
            self._ser.close()


def main() -> int:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    rclpy.init()
    node = RgbVideoRecorder(
        args.topic,
        args.output,
        args.fps,
        args.duration,
        drive=args.drive,
        serial_port=args.port,
        baud_rate=args.baud,
    )
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.close()
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
