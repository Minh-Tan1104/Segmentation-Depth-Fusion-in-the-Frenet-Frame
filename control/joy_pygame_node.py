"""Đọc tay cầm bằng pygame (giống Encoder/wireless.py) và publish sensor_msgs/Joy.

Thay thế joy_node (package "joy", dùng driver evdev/joydev của Linux) vì pygame
đọc tay cầm "Wireless Controller" ổn định hơn trên máy này. Publish ra cùng
topic /joy với cùng định dạng Joy message nên control_node không cần đổi gì
ngoài lại đúng axis/button index của pygame (xem Encoder/wireless.py:
AXIS_SPEED=1, AXIS_STEER=2).
"""
from __future__ import annotations

import pygame
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy


class JoyPygameNode(Node):
    def __init__(self) -> None:
        super().__init__("joy_pygame_node")

        self.declare_parameter("joy_topic", "/joy")
        self.declare_parameter("poll_rate_hz", 50.0)
        self.declare_parameter("joystick_index", 0)

        joy_topic = self.get_parameter("joy_topic").value
        poll_rate_hz = float(self.get_parameter("poll_rate_hz").value)
        joystick_index = int(self.get_parameter("joystick_index").value)

        pygame.init()
        pygame.joystick.init()

        if pygame.joystick.get_count() <= joystick_index:
            raise RuntimeError(
                f"Không tìm thấy tay cầm tại index {joystick_index} "
                f"(tổng số tay cầm pygame thấy: {pygame.joystick.get_count()})"
            )

        self._js = pygame.joystick.Joystick(joystick_index)
        self._js.init()
        self.get_logger().info(
            f"Đã nhận diện tay cầm (pygame): {self._js.get_name()} "
            f"axes={self._js.get_numaxes()} buttons={self._js.get_numbuttons()}"
        )

        self._pub = self.create_publisher(Joy, joy_topic, 10)
        self.create_timer(1.0 / poll_rate_hz, self._tick)

        self.get_logger().info("joy_pygame_node ready")

    def _tick(self) -> None:
        pygame.event.pump()

        msg = Joy()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.axes = [float(self._js.get_axis(i)) for i in range(self._js.get_numaxes())]
        msg.buttons = [int(self._js.get_button(i)) for i in range(self._js.get_numbuttons())]
        self._pub.publish(msg)


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = JoyPygameNode()
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
