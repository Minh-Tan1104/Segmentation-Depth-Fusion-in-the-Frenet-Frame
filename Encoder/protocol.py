"""Hoverboard serial wire protocol.

Cùng giao thức với speed_control.py (struct '<HhhH' gửi, '<HhhhhhhHH' nhận),
chỉ khác là parse_feedback() gọi callback thay vì in ra màn hình, để node ROS2
dùng được trực tiếp.
"""
from __future__ import annotations

import struct
from typing import Callable

START_FRAME = 0xABCD
SPEED_DIVISOR = 16.0


def send_command(ser, steer: int, speed: int) -> None:
    """Đóng gói và gửi lệnh điều khiển xuống Hoverboard.

    Cấu trúc C++: uint16_t start, int16_t steer, int16_t speed, uint16_t checksum
    """
    checksum = (START_FRAME ^ (steer & 0xFFFF) ^ (speed & 0xFFFF)) & 0xFFFF
    packet = struct.pack('<HhhH', START_FRAME, steer, speed, checksum)
    ser.write(packet)


def decode_feedback_frame(frame: bytes) -> dict | None:
    """Giải mã 1 khung feedback 18 bytes từ Hoverboard.

    Trả về dict (đơn vị đã quy đổi: V, °C, RPM) hoặc None nếu checksum sai/lỗi.
    """
    try:
        unpacked = struct.unpack('<HhhhhhhHH', frame)
        start, cmd1, cmd2, speedR, speedL, batVolt, temp, led, checksum = unpacked

        calc_check = (start ^ (cmd1 & 0xFFFF) ^ (cmd2 & 0xFFFF) ^
                      (speedR & 0xFFFF) ^ (speedL & 0xFFFF) ^
                      (batVolt & 0xFFFF) ^ (temp & 0xFFFF) ^ (led & 0xFFFF)) & 0xFFFF

        if checksum != calc_check:
            return None

        return {
            'batVolt': batVolt / 100.0,
            'temp': temp / 10.0,
            'speedL_rpm': speedL / SPEED_DIVISOR,
            'speedR_rpm': speedR / SPEED_DIVISOR,
        }
    except Exception:
        return None


def parse_feedback(buffer: bytes, on_frame: Callable[[dict], None]) -> bytes:
    """Tìm và giải mã các khung feedback 18 bytes trong buffer.

    Gọi on_frame(decoded) cho mỗi khung hợp lệ. Trả về phần buffer còn lại
    (chưa đủ 18 bytes hoặc chưa tìm thấy start frame).
    """
    while len(buffer) >= 18:
        if buffer[0] == 0xCD and buffer[1] == 0xAB:
            frame = buffer[:18]
            buffer = buffer[18:]
            decoded = decode_feedback_frame(frame)
            if decoded is not None:
                on_frame(decoded)
        else:
            buffer = buffer[1:]
    return buffer
