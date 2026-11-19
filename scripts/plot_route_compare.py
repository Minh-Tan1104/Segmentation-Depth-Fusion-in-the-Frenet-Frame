#!/usr/bin/env python3
"""Vẽ đồ thị so sánh 3 nguồn ở TẦNG ROUTE (GPS/RouteEKF): GPS raw, encoder
thuần (route_ekf_enc, không bao giờ correct_gps), EKF fused (route_ekf thật).

CSV do gps_node ghi khi bật log_latlon_enable (xem Gps/node.py: _tick(),
writerow trong khối "Ghi CSV trực tiếp"). Cột dùng ở đây:
  t_s, in_curve_zone, s_m, d_m, sigma_d          (EKF fused)
  raw_lat, raw_lon, raw_fix_type, raw_h_acc_m    (GPS thô, mẫu mới nhất)
  enc_s_m, enc_d_m, enc_sigma_d                  (encoder thuần)

raw_lat/raw_lon là toạ độ WGS84 -> phải chiếu lên tuyến CSV (matcher.project_xy)
mới ra (raw_s_m, raw_d_m) so được với 2 cột kia. Vì GPS cập nhật chậm hơn tick
(50Hz) nên raw_lat/raw_lon lặp lại giữa 2 lần fix — bình thường, không phải lỗi.

Mục đích: thấy encoder thuần trôi bao nhiêu nếu KHÔNG có GPS sửa, và GPS thô
nhiễu/lệch bao nhiêu so với EKF fused — đặc biệt XUYÊN QUA curve zone (khác
plot_dheading_compare.py chỉ xét đoạn thẳng ở tầng FrenetEKF).

Thuần Python + matplotlib + pyproj (qua map_matcher), KHÔNG import rclpy.

Dùng:
  python3 scripts/plot_route_compare.py ~/rl_car_run_20260721_101530.csv
  python3 scripts/plot_route_compare.py <csv> --out compare_route.png
  python3 scripts/plot_route_compare.py <csv> --route-csv map/gps_log.csv
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

# Cho phép chạy trực tiếp `python3 scripts/plot_route_compare.py` mà không
# cần cài package qua colcon — chèn root RL_CAR (cha của scripts/) vào
# sys.path để import Gps.map_matcher như module thường.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from Gps.map_matcher import RouteMapMatcher  # noqa: E402


def load_csv(path: str) -> dict[str, list]:
    cols: dict[str, list] = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise SystemExit(f"CSV rong hoac khong co header: {path}")
        for name in reader.fieldnames:
            cols[name] = []
        for row in reader:
            for name in reader.fieldnames:
                cols[name].append(row.get(name, ""))
    return cols


def to_float(vals: list[str]) -> list[float]:
    out = []
    for v in vals:
        try:
            out.append(float(v))
        except (ValueError, TypeError):
            out.append(float("nan"))
    return out


def rmse(a: list[float], b: list[float]) -> float:
    se = [(x - y) ** 2 for x, y in zip(a, b) if not (math.isnan(x) or math.isnan(y))]
    return math.sqrt(sum(se) / len(se)) if se else float("nan")


def find_default_route_csv() -> Path:
    return Path(__file__).resolve().parent.parent / "map" / "gps_log.csv"


def main() -> None:
    ap = argparse.ArgumentParser(description="Ve do thi so sanh GPS raw / encoder thuan / EKF fused (tang route).")
    ap.add_argument("csv", help="File CSV do gps_node ghi (log_latlon_enable).")
    ap.add_argument("--route-csv", default=None, help="Tuyen CSV (mac dinh map/gps_log.csv).")
    ap.add_argument("--out", default=None, help="Luu PNG thay vi mo cua so (vd compare_route.png).")
    args = ap.parse_args()

    try:
        import matplotlib
        if args.out:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        raise SystemExit("Thieu matplotlib. Cai: pip install matplotlib")

    cols = load_csv(args.csv)
    for need in ("t_s", "in_curve_zone", "s_m", "d_m", "sigma_d",
                 "raw_lat", "raw_lon", "enc_s_m", "enc_d_m"):
        if need not in cols:
            raise SystemExit(
                f"CSV thieu cot '{need}' — co dung file gps_node ghi (log_latlon_enable ban moi) khong?"
            )
    if not cols["t_s"]:
        raise SystemExit("CSV khong co dong du lieu nao.")

    t = to_float(cols["t_s"])
    in_zone = to_float(cols["in_curve_zone"])
    ekf_s, ekf_d, ekf_sigma = to_float(cols["s_m"]), to_float(cols["d_m"]), to_float(cols["sigma_d"])
    enc_s, enc_d = to_float(cols["enc_s_m"]), to_float(cols["enc_d_m"])

    route_csv = Path(args.route_csv) if args.route_csv else find_default_route_csv()
    if not route_csv.exists():
        raise SystemExit(f"Khong tim thay tuyen CSV: {route_csv}")
    matcher = RouteMapMatcher(route_csv)

    # Chiếu raw_lat/raw_lon -> (raw_s_m, raw_d_m) so được với 2 track kia.
    # Chỉ chiếu các dòng CÓ raw fix (bỏ qua "" trước khi có fix đầu tiên).
    raw_t, raw_s, raw_d = [], [], []
    for ti, lat_s, lon_s in zip(t, cols["raw_lat"], cols["raw_lon"]):
        if not lat_s or not lon_s:
            continue
        try:
            lat, lon = float(lat_s), float(lon_s)
        except ValueError:
            continue
        x, y = matcher.to_local(lat, lon)
        s_m, d_m, _seg = matcher.project_xy(x, y)
        raw_t.append(ti)
        raw_s.append(s_m)
        raw_d.append(d_m)

    # RMSE: EKF fused coi như tham chiếu tốt nhất hiện có (encoder+GPS+anchor).
    enc_rmse_d = rmse(enc_d, ekf_d)
    enc_rmse_s = rmse(enc_s, ekf_s)
    # raw không cùng lưới thời gian với ekf (ít điểm hơn) -> nội suy ekf_d/ekf_s
    # về đúng thời điểm raw_t rồi mới so.
    import numpy as np
    ekf_d_at_raw = np.interp(raw_t, t, ekf_d).tolist() if raw_t else []
    ekf_s_at_raw = np.interp(raw_t, t, ekf_s).tolist() if raw_t else []
    raw_rmse_d = rmse(raw_d, ekf_d_at_raw)
    raw_rmse_s = rmse(raw_s, ekf_s_at_raw)

    # Vùng curve zone (in_curve_zone chuyển 0->1/1->0) để tô nền, xuyên suốt
    # cả 2 subplot -> dễ đối chiếu thời điểm encoder/raw lệch nhiều có đúng
    # vào lúc GPS bị chặn (trong zone) hay không.
    zone_spans: list[tuple[float, float]] = []
    z_start = None
    for ti, z in zip(t, in_zone):
        if z >= 0.5 and z_start is None:
            z_start = ti
        elif z < 0.5 and z_start is not None:
            zone_spans.append((z_start, ti))
            z_start = None
    if z_start is not None:
        zone_spans.append((z_start, t[-1]))

    fig, (ax_d, ax_s) = plt.subplots(2, 1, figsize=(12, 7.5), sharex=True)

    for ax in (ax_d, ax_s):
        for a, b in zone_spans:
            ax.axvspan(a, b, color="orange", alpha=0.12, zorder=0, label="_nolegend_")

    ax_d.plot(t, ekf_d, color="#1f77b4", lw=1.8,
              label=f"EKF fused (d, sigma_d trung binh={sum(ekf_sigma)/len(ekf_sigma):.3f}m)")
    ax_d.plot(t, enc_d, color="#d62728", lw=1.4, ls="--",
              label=f"encoder thuan (RMSE vs EKF={enc_rmse_d:.3f}m)")
    ax_d.scatter(raw_t, raw_d, color="#2ca02c", s=10, zorder=3,
                 label=f"GPS raw chieu len tuyen (RMSE vs EKF={raw_rmse_d:.3f}m)")
    ax_d.axhline(0.0, color="0.7", lw=0.8, zorder=0)
    ax_d.set_ylabel("d [m]  (+ = phai tuyen)")
    ax_d.set_title("So sanh lech ngang d — GPS raw / encoder thuan / EKF fused (nen cam = curve zone)")
    ax_d.grid(True, alpha=0.3)
    ax_d.legend(loc="best", fontsize=9)

    ax_s.plot(t, ekf_s, color="#1f77b4", lw=1.8, label="EKF fused (s)")
    ax_s.plot(t, enc_s, color="#d62728", lw=1.4, ls="--",
              label=f"encoder thuan (RMSE vs EKF={enc_rmse_s:.3f}m)")
    ax_s.scatter(raw_t, raw_s, color="#2ca02c", s=10, zorder=3,
                 label=f"GPS raw chieu len tuyen (RMSE vs EKF={raw_rmse_s:.3f}m)")
    ax_s.set_ylabel("s [m]  (doc tuyen)")
    ax_s.set_xlabel("thoi gian [s]")
    ax_s.set_title("So sanh vi tri doc tuyen s")
    ax_s.grid(True, alpha=0.3)
    ax_s.legend(loc="best", fontsize=9)

    fig.tight_layout()

    print(f"Doc {len(t)} dong tu {args.csv} ({len(raw_t)} dong co GPS fix, {len(zone_spans)} curve zone)")
    print(f"  d RMSE vs EKF fused:  encoder={enc_rmse_d:.4f} m | GPS raw={raw_rmse_d:.4f} m")
    print(f"  s RMSE vs EKF fused:  encoder={enc_rmse_s:.4f} m | GPS raw={raw_rmse_s:.4f} m")

    if args.out:
        fig.savefig(args.out, dpi=130)
        print(f"Da luu do thi -> {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
