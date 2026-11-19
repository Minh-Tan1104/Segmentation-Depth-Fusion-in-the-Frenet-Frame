import serial
import struct
import time
import sys
import pygame

# ================== CẤU HÌNH GIAO TIẾP SERIAL ==================
PORT = '/dev/ttyUSB0'         # Đổi thành cổng của bạn ('COM3' trên Windows)
BAUD_RATE = 115200
TIME_SEND = 0.05              # 50ms - tương đương TIME_SEND bên Arduino
START_FRAME = 0xABCD

# ================== CẤU HÌNH TỐC ĐỘ ==================
MAX_SPEED = 150             # Tốc độ tối đa khi đẩy cần hết cỡ (tiến/lùi)
MAX_STEER = 50              # Tốc độ rẽ tối đa
SPEED_DIVISOR = 16.0           # Quy đổi speed thô của firmware Hoverboard về RPM
DEADZONE = 0.15                # Vùng chết của cần analog (0.0 - 1.0)

# ================== CẤU HÌNH TAY CẦM PS4 ==================
# CHẠY TRƯỚC: python3 ps4_hoverboard_control.py --test
# để xem đúng chỉ số axis/button trên thiết bị Jetson của bạn,
# rồi chỉnh lại 3 hằng số dưới đây cho khớp.
AXIS_SPEED = 1      # Trục dọc cần trái (Left stick Y) -> tiến/lùi
AXIS_STEER = 2      # Trục ngang cần phải (Right stick X) -> rẽ trái/phải (yaw). Đổi số này nếu sai, dùng --test để dò
BUTTON_EXIT = 9     # Nút "Options" (tuỳ driver) -> thoát chương trình


def send_command(ser, steer, speed):
    """
    Đóng gói và gửi lệnh điều khiển xuống Hoverboard.
    Cấu trúc C++: uint16_t start, int16_t steer, int16_t speed, uint16_t checksum
    """
    checksum = (START_FRAME ^ (steer & 0xFFFF) ^ (speed & 0xFFFF)) & 0xFFFF
    packet = struct.pack('<HhhH', START_FRAME, steer, speed, checksum)
    ser.write(packet)


def decode_feedback_frame(frame):
    """
    Giải mã 1 khung feedback 18 bytes từ Hoverboard.
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
    except Exception as e:
        print(f"\n[Lỗi giải mã]: {e}")
        return None


def parse_feedback(buffer):
    """
    Tìm, giải mã và in ra màn hình khung dữ liệu phản hồi từ Hoverboard.
    Trả về buffer còn lại.
    """
    while len(buffer) >= 18:
        if buffer[0] == 0xCD and buffer[1] == 0xAB:
            frame = buffer[:18]
            buffer = buffer[18:]

            decoded = decode_feedback_frame(frame)
            if decoded is not None:
                print(f"\r[Xe] Pin: {decoded['batVolt']:.2f}V | Nhiệt độ: {decoded['temp']:.1f}°C | "
                      f"Bánh Trái: {decoded['speedL_rpm']:.2f} | Bánh Phải: {decoded['speedR_rpm']:.2f}  ", end="")
        else:
            buffer = buffer[1:]

    return buffer


def apply_deadzone(value, deadzone=DEADZONE):
    """
    Loại bỏ nhiễu nhỏ quanh vị trí trung tâm của cần analog,
    đồng thời rescale lại để vẫn dùng được full dải -1.0 .. 1.0
    ngay sau khi vượt qua vùng chết (tránh giật cục khi vừa rời tâm).
    """
    if abs(value) < deadzone:
        return 0.0
    sign = 1.0 if value > 0 else -1.0
    return sign * (abs(value) - deadzone) / (1.0 - deadzone)


def test_mode():
    """
    Chế độ kiểm tra: in ra toàn bộ giá trị axis/button của tay cầm theo thời gian thực,
    dùng để xác định đúng chỉ số AXIS_SPEED / AXIS_STEER / BUTTON_EXIT trước khi điều khiển thật.
    """
    pygame.init()
    pygame.joystick.init()

    if pygame.joystick.get_count() == 0:
        print("Không tìm thấy tay cầm nào. Kiểm tra kết nối USB/Bluetooth rồi thử lại.")
        return

    js = pygame.joystick.Joystick(0)
    js.init()
    print(f"Đã nhận diện: {js.get_name()}")
    print(f"Số trục (axes): {js.get_numaxes()} | Số nút (buttons): {js.get_numbuttons()}")
    print("Di chuyển cần và bấm các nút để xem chỉ số tương ứng. Ctrl+C để thoát.\n")

    try:
        while True:
            pygame.event.pump()
            axes = [round(js.get_axis(i), 2) for i in range(js.get_numaxes())]
            buttons = [js.get_button(i) for i in range(js.get_numbuttons())]
            print(f"\rAxes: {axes}  Buttons: {buttons}   ", end="")
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("\nThoát chế độ test.")


def main():
    try:
        ser = serial.Serial(PORT, BAUD_RATE, timeout=0.01)
        print(f"Đã kết nối thành công tới {PORT} ở tốc độ {BAUD_RATE}")
    except Exception as e:
        print(f"Lỗi mở cổng Serial: {e}")
        return

    pygame.init()
    pygame.joystick.init()

    if pygame.joystick.get_count() == 0:
        print("Không tìm thấy tay cầm PS4. Hãy cắm USB hoặc ghép nối Bluetooth trước khi chạy.")
        ser.close()
        return

    js = pygame.joystick.Joystick(0)
    js.init()
    print(f"Đã nhận diện tay cầm: {js.get_name()}")

    print("=== ĐIỀU KHIỂN HOVERBOARD BẰNG TAY CẦM PS4 ===")
    print(" - Cần trái lên/xuống  : TIẾN / LÙI")
    print(" - Cần phải trái/phải  : RẼ TRÁI / RẼ PHẢI (yaw)")
    print(f" - Nút Options (index {BUTTON_EXIT}) hoặc Ctrl+C : THOÁT\n")

    last_send_time = 0
    rx_buffer = b''

    try:
        while True:
            # 1. ĐỌC TÍN HIỆU PHẢN HỒI TỪ HOVERBOARD
            if ser.in_waiting > 0:
                rx_buffer += ser.read(ser.in_waiting)
                rx_buffer = parse_feedback(rx_buffer)

            # 2. CẬP NHẬT TRẠNG THÁI TAY CẦM
            pygame.event.pump()

            if js.get_numbuttons() > BUTTON_EXIT and js.get_button(BUTTON_EXIT):
                print("\n\nĐã nhấn nút thoát...")
                break

            # Đẩy cần lên thường trả về giá trị âm -> đảo dấu để "lên = tiến"
            raw_speed_axis = -js.get_axis(AXIS_SPEED)
            raw_steer_axis = js.get_axis(AXIS_STEER)

            speed_norm = apply_deadzone(raw_speed_axis)
            steer_norm = apply_deadzone(raw_steer_axis)

            speed = int(speed_norm * MAX_SPEED)
            steer = int(steer_norm * MAX_STEER)

            # 3. GỬI LỆNH (mỗi 50ms, giữ nguyên nhịp như bản gốc)
            current_time = time.time()
            if current_time - last_send_time >= TIME_SEND:
                send_command(ser, steer, speed)
                last_send_time = current_time

            time.sleep(0.005)

    except KeyboardInterrupt:
        print("\n\nĐã dừng đột ngột!")
    finally:
        send_command(ser, 0, 0)  # Dừng hẳn xe trước khi thoát, đảm bảo an toàn
        ser.close()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        test_mode()
    else:
        main()