import serial
import struct
import time
import keyboard  # Thư viện bắt sự kiện bàn phím

# --- CẤU HÌNH GIAO TIẾP ---
PORT = '/dev/ttyUSB0'         # Đổi thành cổng của bạn (VD: 'COM3' trên Windows, hoặc '/dev/ttyUSB0' trên Linux/Orange Pi)
BAUD_RATE = 115200
TIME_SEND = 0.05      # 50ms - Tương đương TIME_SEND trong Arduino
START_FRAME = 0xABCD

# --- CẤU HÌNH TỐC ĐỘ ---
MAX_SPEED = 50       # Tốc độ chạy thẳng (W, S)
MAX_STEER = 150       # Tốc độ rẽ (A, D)
SPEED_DIVISOR = 16.0  # Hệ số quy đổi speed thô của firmware Hoverboard về RPM

def send_command(ser, steer, speed):
    """
    Đóng gói và gửi lệnh điều khiển xuống Hoverboard.
    Cấu trúc C++: uint16_t start, int16_t steer, int16_t speed, uint16_t checksum
    """
    # Ép kiểu bitwise (masking & 0xFFFF) để mô phỏng kiểu uint16_t của C++ trong Python
    checksum = (START_FRAME ^ (steer & 0xFFFF) ^ (speed & 0xFFFF)) & 0xFFFF
    
    # '<HhhH' nghĩa là:
    # < : Little-endian (chuẩn của vi điều khiển)
    # H : unsigned short (uint16_t) -> start
    # h : short (int16_t)          -> steer
    # h : short (int16_t)          -> speed
    # H : unsigned short (uint16_t) -> checksum
    packet = struct.pack('<HhhH', START_FRAME, steer, speed, checksum)
    
    ser.write(packet)

def decode_feedback_frame(frame):
    """
    Giải mã 1 khung feedback 18 bytes từ Hoverboard.
    Trả về dict (đơn vị đã quy đổi: V, °C, RPM) hoặc None nếu checksum sai/lỗi.
    """
    try:
        # SỬA Ở ĐÂY: <H (1), h (6), HH (2) -> Tổng 9 biến, 18 bytes
        unpacked = struct.unpack('<HhhhhhhHH', frame)
        start, cmd1, cmd2, speedR, speedL, batVolt, temp, led, checksum = unpacked

        # Tính toán lại checksum để kiểm tra nhiễu
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
        # SỬA Ở ĐÂY: In lỗi ra màn hình thay vì 'pass' để dễ debug
        print(f"\n[Lỗi giải mã]: {e}")
        return None

def parse_feedback(buffer):
    """
    Tìm, giải mã và in ra màn hình khung dữ liệu phản hồi từ Hoverboard.
    Trả về buffer_còn_lại.
    """
    # Gói tin Feedback dài đúng 18 bytes
    while len(buffer) >= 18:
        # Kiểm tra Start Frame (0xABCD ở Little-endian sẽ là 0xCD, 0xAB)
        if buffer[0] == 0xCD and buffer[1] == 0xAB:
            frame = buffer[:18]
            buffer = buffer[18:] # Cắt bỏ 18 bytes đã đọc

            decoded = decode_feedback_frame(frame)
            if decoded is not None:
                # In ra các thông số quan trọng
                print(f"\r[Xe] Pin: {decoded['batVolt']:.2f}V | Nhiệt độ: {decoded['temp']:.1f}°C | "
                      f"Bánh Trái: {decoded['speedL_rpm']:.2f} | Bánh Phải: {decoded['speedR_rpm']:.2f}  ", end="")
        else:
            # Nếu byte đầu không khớp, dịch đi 1 byte để tìm Start Frame mới
            buffer = buffer[1:]

    return buffer

def main():
    try:
        # Mở cổng Serial
        ser = serial.Serial(PORT, BAUD_RATE, timeout=0.01)
        print(f" Đã kết nối thành công tới {PORT} ở tốc độ {BAUD_RATE}")
    except Exception as e:
        print(f" Lỗi mở cổng Serial: {e}")
        return

    print("=== ĐIỀU KHIỂN HOVERBOARD BẰNG BÀN PHÍM ===")
    print(" - Nhấn và giữ phím 'W' để TIẾN")
    print(" - Nhấn và giữ phím 'S' để LÙI")
    print(" - Nhấn và giữ phím 'A' để RẼ TRÁI")
    print(" - Nhấn và giữ phím 'D' để RẼ PHẢI")
    print(" - Nhấn 'ESC' để thoát chương trình\n")

    last_send_time = 0
    rx_buffer = b''

    try:
        while True:
            # 1. ĐỌC TÍN HIỆU TỪ HOVERBOARD
            if ser.in_waiting > 0:
                rx_buffer += ser.read(ser.in_waiting)
                rx_buffer = parse_feedback(rx_buffer)

            # 2. XỬ LÝ PHÍM BẤM BẰNG AWDS
            if keyboard.is_pressed('esc'):
                print("\n\n Đang thoát...")
                break

            speed = 0
            steer = 0

            if keyboard.is_pressed('w'):
                speed = MAX_SPEED
            elif keyboard.is_pressed('s'):
                speed = -MAX_SPEED

            if keyboard.is_pressed('a'):
                steer = -MAX_STEER
            elif keyboard.is_pressed('d'):
                steer = MAX_STEER

            # 3. GỬI LỆNH (Mỗi 50ms)
            current_time = time.time()
            if current_time - last_send_time >= TIME_SEND:
                send_command(ser, steer, speed)
                last_send_time = current_time
                
            # Nghỉ một chút để tránh vắt kiệt CPU
            time.sleep(0.005)

    except KeyboardInterrupt:
        print("\n\n Đã dừng đột ngột!")
    finally:
        # Trước khi đóng cổng, gửi lệnh dừng hẳn xe lại cho an toàn
        send_command(ser, 0, 0)
        ser.close()

if __name__ == "__main__":
    main()