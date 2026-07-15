import math
import time

import keyboard
import matplotlib.pyplot as plt
import serial

from control_speed import (
    BAUD_RATE,
    MAX_SPEED,
    MAX_STEER,
    PORT,
    TIME_SEND,
    decode_feedback_frame,
    send_command,
)

# --- CẤU HÌNH ROBOT (differential drive) ---
WHEEL_RADIUS = 0.0762   # m (bán kính bánh xe, 3 in)
TRACK_WIDTH = 0.3556    # m (khoảng cách giữa 2 bánh, 14 in)
RPM_TO_MPS = 2 * math.pi * WHEEL_RADIUS / 60.0  # quy đổi RPM -> m/s
RIGHT_WHEEL_SIGN = -1  # Encoder bánh phải bị ngược dấu so với bánh trái khi lắp đặt

PLOT_PERIOD = 0.1  # 10Hz, tần số vẽ lại plot


def parse_feedback_to_odom(buffer, odom):
    """
    Giống parse_feedback trong control_speed.py nhưng cập nhật vận tốc bánh
    vào dict `odom` thay vì in ra màn hình. Trả về buffer còn lại.
    """
    while len(buffer) >= 18:
        if buffer[0] == 0xCD and buffer[1] == 0xAB:
            frame = buffer[:18]
            buffer = buffer[18:]

            decoded = decode_feedback_frame(frame)
            if decoded is not None:
                odom['speedL_rpm'] = decoded['speedL_rpm']
                odom['speedR_rpm'] = RIGHT_WHEEL_SIGN * decoded['speedR_rpm']
        else:
            buffer = buffer[1:]

    return buffer


def integrate_odometry(odom, dt):
    """Cập nhật x, y, theta theo mô hình differential drive."""
    vL = odom['speedL_rpm'] * RPM_TO_MPS
    vR = odom['speedR_rpm'] * RPM_TO_MPS

    v = (vL + vR) / 2.0
    w = (vR - vL) / TRACK_WIDTH

    odom['theta'] += w * dt
    odom['x'] += v * math.cos(odom['theta']) * dt
    odom['y'] += v * math.sin(odom['theta']) * dt


def main():
    try:
        ser = serial.Serial(PORT, BAUD_RATE, timeout=0.01)
        print(f" Đã kết nối thành công tới {PORT} ở tốc độ {BAUD_RATE}")
    except Exception as e:
        print(f" Lỗi mở cổng Serial: {e}")
        return

    print("=== ODOMETRY REALTIME TỪ ENCODER HOVERBOARD ===")
    print(" - Nhấn và giữ W/S/A/D để điều khiển xe")
    print(" - Nhấn 'ESC' để thoát chương trình\n")

    odom = {'x': 0.0, 'y': 0.0, 'theta': 0.0, 'speedL_rpm': 0.0, 'speedR_rpm': 0.0}
    rx_buffer = b''

    plt.ion()
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('Quỹ đạo Odometry (Encoder Hoverboard)')
    ax.grid(True)
    path_line, = ax.plot([0.0], [0.0], 'b-', linewidth=1.5)
    heading_arrow = ax.quiver([0.0], [0.0], [1.0], [0.0], color='r', scale=20)
    path_x, path_y = [0.0], [0.0]

    last_send_time = 0.0
    last_plot_time = 0.0
    last_odom_time = time.time()

    try:
        while True:
            # 1. ĐỌC VÀ GIẢI MÃ TÍN HIỆU TỪ HOVERBOARD
            if ser.in_waiting > 0:
                rx_buffer += ser.read(ser.in_waiting)
                rx_buffer = parse_feedback_to_odom(rx_buffer, odom)

            # 2. TÍCH PHÂN ODOMETRY
            now = time.time()
            dt = now - last_odom_time
            last_odom_time = now
            integrate_odometry(odom, dt)

            # 3. XỬ LÝ PHÍM BẤM WASD
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

            # 4. GỬI LỆNH (mỗi 50ms)
            if now - last_send_time >= TIME_SEND:
                send_command(ser, steer, speed)
                last_send_time = now

            # 5. VẼ LẠI PLOT (10Hz)
            if now - last_plot_time >= PLOT_PERIOD:
                path_x.append(odom['x'])
                path_y.append(odom['y'])
                path_line.set_data(path_x, path_y)
                ax.relim()
                ax.autoscale_view()
                heading_arrow.set_offsets([[odom['x'], odom['y']]])
                heading_arrow.set_UVC([math.cos(odom['theta'])], [math.sin(odom['theta'])])
                fig.canvas.draw_idle()
                last_plot_time = now

            plt.pause(0.001)

    except KeyboardInterrupt:
        print("\n\n Đã dừng đột ngột!")
    finally:
        # Trước khi đóng cổng, gửi lệnh dừng hẳn xe lại cho an toàn
        send_command(ser, 0, 0)
        ser.close()
        plt.ioff()
        plt.show()


if __name__ == "__main__":
    main()
