#!/usr/bin/env python3
"""Đọc GPS đơn giản: in lat/lon/fix/sats/hAcc từ module u-blox ra terminal.

Standalone, KHÔNG cần ROS/rclpy — chỉ cần pyserial + pyubx2. Dùng để kiểm tra
nhanh module GPS có fix chưa, toạ độ đang ở đâu, hAcc bao nhiêu, trước khi chạy
cả stack. Cùng cách parse UBX NAV-PVT với Gps/gps_reader.py trong package.

Cài phụ thuộc (nếu thiếu):
    pip install pyserial pyubx2

Chạy:
    python3 read_gps.py                    # dùng PORT/BAUD mặc định bên dưới
    python3 read_gps.py /dev/ttyUSB1       # chỉ định cổng
    python3 read_gps.py /dev/ttyUSB1 230400
    python3 read_gps.py --csv gps_log.csv  # vừa in vừa ghi CSV (lat,lon,...)

Ctrl+C để thoát.
"""
import argparse
import csv
import sys
import time

try:
    from serial import Serial, SerialException
    from pyubx2 import UBXReader
except ImportError as exc:
    sys.exit(
        f"Thieu thu vien '{exc.name}'. Cai bang: pip install pyserial pyubx2"
    )

# ================== CẤU HÌNH (khớp gps_node trong rl_car_params.yaml) ==================
PORT = "/dev/ttyUSB1"   # cổng module GPS (encoder dùng /dev/ttyUSB0)
BAUD = 230400           # baudrate UBX
TIMEOUT = 1.0           # timeout đọc serial [s]

FIX_LABELS = {
    0: "no-fix", 1: "dead-reckon", 2: "2D", 3: "3D", 4: "GNSS+DR", 5: "time-only",
}


def main() -> None:
    ap = argparse.ArgumentParser(description="Doc GPS u-blox, in lat/lon/fix/sats/hAcc.")
    ap.add_argument("port", nargs="?", default=PORT, help=f"Cong serial (mac dinh {PORT})")
    ap.add_argument("baud", nargs="?", type=int, default=BAUD, help=f"Baudrate (mac dinh {BAUD})")
    ap.add_argument("--csv", default=None, help="Ghi them ra file CSV (lat,lon,fix,sats,hAcc_m).")
    args = ap.parse_args()

    try:
        ser = Serial(args.port, args.baud, timeout=TIMEOUT)
    except SerialException as exc:
        sys.exit(f"Khong mo duoc cong {args.port} @ {args.baud}: {exc}")
    print(f"Da mo {args.port} @ {args.baud}. Cho ban tin NAV-PVT... (Ctrl+C de thoat)\n")

    writer = None
    csv_file = None
    if args.csv:
        csv_file = open(args.csv, "w", newline="")
        writer = csv.writer(csv_file)
        writer.writerow(["t_s", "lat", "lon", "fix_type", "num_sv", "h_acc_m"])
        print(f"Ghi CSV -> {args.csv}\n")

    reader = UBXReader(ser)
    t0 = time.monotonic()
    n = 0
    try:
        while True:
            try:
                _raw, msg = reader.read()
            except Exception as exc:  # lỗi parse thoáng qua -> bỏ qua, đọc tiếp
                print(f"\n[parse loi, bo qua]: {exc}")
                continue
            if msg is None:
                continue
            # Chỉ quan tâm NAV-PVT (có lat/lon). Bỏ qua bản tin khác.
            lat = getattr(msg, "lat", None)
            lon = getattr(msg, "lon", None)
            if lat is None or lon is None:
                continue

            fix = getattr(msg, "fixType", None)
            sats = getattr(msg, "numSV", None)
            h_acc_mm = getattr(msg, "hAcc", None)          # NAV-PVT hAcc theo mm
            h_acc_m = None if h_acc_mm is None else float(h_acc_mm) * 1e-3
            fix_lbl = FIX_LABELS.get(fix, str(fix))

            n += 1
            h_acc_str = "?" if h_acc_m is None else f"{h_acc_m:.2f}m"
            print(
                f"\r#{n:<5} lat={lat:.7f} lon={lon:.7f} | fix={fix_lbl:<10} "
                f"sats={sats if sats is not None else '?':<3} hAcc={h_acc_str:<8}",
                end="",
                flush=True,
            )
            if writer is not None:
                writer.writerow([
                    f"{time.monotonic() - t0:.3f}", f"{lat:.7f}", f"{lon:.7f}",
                    fix if fix is not None else "",
                    sats if sats is not None else "",
                    "" if h_acc_m is None else f"{h_acc_m:.3f}",
                ])
                csv_file.flush()
    except KeyboardInterrupt:
        print("\n\nDa thoat.")
    finally:
        ser.close()
        if csv_file is not None:
            csv_file.close()


if __name__ == "__main__":
    main()
