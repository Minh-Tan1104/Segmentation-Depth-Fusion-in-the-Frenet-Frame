#!/usr/bin/env python3
"""Đọc GPS u-blox, in GỌN: sats + lat/lon cho mỗi NAV-PVT (kể cả chưa có fix).

Bản rút gọn của viewer cũ — bỏ matplotlib/contextily/RXM-RAWX, chỉ cần
pyserial + pyubx2. Dùng để xem nhanh số vệ tinh (numSV) đang bắt được và toạ
độ. LƯU Ý: khi sats=0 / fix=no-fix thì lat/lon là RÁC (u-blox vẫn tự điền sẵn
trường này) — chờ sats tăng (>=4) và fix=3D thì toạ độ mới thật.

Cài (nếu thiếu):
    pip install pyserial pyubx2

Chạy:
    python3 Gps.py                              # /dev/ttyUSB1 @ 230400 mặc định
    python3 Gps.py --port /dev/ttyUSB0 --baud 230400
Ctrl+C để thoát.
"""
import argparse
import sys

try:
    from pyubx2 import UBXReader
    from serial import Serial, SerialException
except ImportError as exc:
    sys.exit(f"Thieu thu vien '{exc.name}'. Cai: pip install pyserial pyubx2")

FIX_LABELS = {0: "no-fix", 1: "dead-reckon", 2: "2D", 3: "3D", 4: "GNSS+DR", 5: "time-only"}


def main() -> None:
    ap = argparse.ArgumentParser(description="In sats + lat/lon tu GPS u-blox.")
    ap.add_argument("--port", default="/dev/ttyUSB1", help="Cong serial (mac dinh /dev/ttyUSB1)")
    ap.add_argument("--baud", type=int, default=230400, help="Baudrate (mac dinh 230400)")
    ap.add_argument("--timeout", type=float, default=1.0, help="Timeout doc serial [s]")
    args = ap.parse_args()

    try:
        ser = Serial(args.port, args.baud, timeout=args.timeout)
    except SerialException as exc:
        sys.exit(f"Khong mo duoc {args.port} @ {args.baud}: {exc}")
    print(f"Doc {args.port} @ {args.baud} ... (Ctrl+C de thoat)")

    reader = UBXReader(ser)
    n = 0
    try:
        while True:
            try:
                _raw, msg = reader.read()
            except Exception:
                continue  # loi parse thoang qua -> doc tiep
            if msg is None:
                continue
            # Chi quan tam NAV-PVT (co lat/lon/numSV); bo qua NAV-DOP/MON-HW... (khong dump).
            lat = getattr(msg, "lat", None)
            lon = getattr(msg, "lon", None)
            if lat is None or lon is None:
                continue
            sats = getattr(msg, "numSV", None)
            fix = getattr(msg, "fixType", None)
            h_acc = getattr(msg, "hAcc", None)  # mm -> m
            n += 1
            h_acc_str = f"{h_acc / 1000.0:.2f}m" if h_acc is not None and h_acc > 0 else "?"
            print(
                f"\r#{n:<6} sats={sats if sats is not None else '?':<3} "
                f"fix={FIX_LABELS.get(fix, fix):<11} "
                f"hAcc={h_acc_str:<7} "
                f"lat={lat:.7f} lon={lon:.7f}",
                end="", flush=True,
            )
    except KeyboardInterrupt:
        print("\nThoat.")
    finally:
        ser.close()


if __name__ == "__main__":
    main()
