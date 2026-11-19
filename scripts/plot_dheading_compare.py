#!/usr/bin/env python3
"""Vẽ đồ thị so sánh (d, heading) 3 nguồn từ CSV do control_node ghi.

CSV được control_node ghi khi bật log_compare_enable (xem
config/rl_car_params.yaml + control/node.py:_maybe_log_compare). Mỗi dòng là
1 thời điểm trên đoạn THẲNG (không curve mode), 3 cặp (d, heading) cùng quy
ước vision (d = d_meters, +d = line/lệch sang phải; heading = độ):
  - EKF fused   (encoder predict + vision correct)
  - vision thuần (đo trực tiếp từ perception)
  - encoder thuần (dead-reckon, 1 FrenetEKF chỉ predict, seed 1 lần từ vision)

Mục đích: thấy encoder thuần TRÔI dần so với vision/EKF, còn EKF fused bám
sát vision — minh hoạ tác dụng của EKF fusion.

Thuần Python + matplotlib, KHÔNG import rclpy — chạy offline trên máy bất kỳ.

Dùng:
  python3 scripts/plot_dheading_compare.py ~/rl_car_dheading_20260719_101530.csv
  python3 scripts/plot_dheading_compare.py <csv> --out compare.png      # lưu PNG, không mở cửa sổ
  python3 scripts/plot_dheading_compare.py <csv> --mask-stale-vision    # ẩn vision khi cờ vision_fresh=0
"""
from __future__ import annotations

import argparse
import csv
import math
import sys


def load_csv(path: str) -> dict[str, list[float]]:
    cols: dict[str, list] = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise SystemExit(f"CSV rong hoac khong co header: {path}")
        for name in reader.fieldnames:
            cols[name] = []
        for row in reader:
            for name in reader.fieldnames:
                val = row.get(name, "")
                if name == "mode":
                    cols[name].append(val)
                else:
                    try:
                        cols[name].append(float(val))
                    except (ValueError, TypeError):
                        cols[name].append(float("nan"))
    return cols


def rmse(a: list[float], b: list[float]) -> float:
    """RMSE giữa 2 chuỗi, bỏ qua cặp có NaN."""
    se = [(x - y) ** 2 for x, y in zip(a, b) if not (math.isnan(x) or math.isnan(y))]
    return math.sqrt(sum(se) / len(se)) if se else float("nan")


def _find_stale_ranges(
    t: list[float], vision_fresh: list[float]
) -> list[tuple[float, float]]:
    """Tim cac khoang lien tuc co vision_fresh < 0.5 (vision mat/cu).
    Tra ve [(t_start, t_end), ...]."""
    ranges: list[tuple[float, float]] = []
    in_stale = False
    start = 0.0
    for ti, fr in zip(t, vision_fresh):
        if fr < 0.5 and not in_stale:
            in_stale = True
            start = ti
        elif fr >= 0.5 and in_stale:
            in_stale = False
            ranges.append((start, ti))
    if in_stale:
        ranges.append((start, t[-1]))
    return ranges


def main() -> None:
    ap = argparse.ArgumentParser(description="Ve do thi so sanh (d, heading) 3 nguon tu CSV.")
    ap.add_argument("csv", help="Duong dan file CSV do control_node ghi.")
    ap.add_argument("--out", default=None, help="Luu PNG thay vi mo cua so (vd compare.png).")
    ap.add_argument("--mask-stale-vision", action="store_true",
                    help="An diem vision khi vision_fresh=0 (vision cu, khong tin).")
    args = ap.parse_args()

    try:
        import matplotlib
        if args.out:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        raise SystemExit("Thieu matplotlib. Cai: pip install matplotlib")

    cols = load_csv(args.csv)
    for need in ("t_s", "ekf_d", "vision_d", "enc_d", "ekf_heading_deg",
                 "vision_heading_deg", "enc_heading_deg"):
        if need not in cols:
            raise SystemExit(f"CSV thieu cot '{need}' — co dung file do control_node ghi khong?")
    if not cols["t_s"]:
        raise SystemExit("CSV khong co dong du lieu nao (co the chua chay tren doan thang / chua bat log).")

    t = cols["t_s"]
    vis_d = list(cols["vision_d"])
    vis_h = list(cols["vision_heading_deg"])

    # Phat hien khoang mat vision (dung de to nen).
    has_fresh_col = "vision_fresh" in cols
    if has_fresh_col:
        stale_ranges = _find_stale_ranges(t, cols["vision_fresh"])
    else:
        # Fallback: vision mat = NaN trong vision_d.
        fresh = [0.0 if math.isnan(d) else 1.0 for d in vis_d]
        stale_ranges = _find_stale_ranges(t, fresh)

    if args.mask_stale_vision and has_fresh_col:
        vis_d = [d if fr >= 0.5 else float("nan") for d, fr in zip(vis_d, cols["vision_fresh"])]
        vis_h = [h if fr >= 0.5 else float("nan") for h, fr in zip(vis_h, cols["vision_fresh"])]

    # RMSE so với vision (vision coi như tham chiếu đo trực tiếp làn).
    ekf_rmse_d = rmse(cols["ekf_d"], vis_d)
    enc_rmse_d = rmse(cols["enc_d"], vis_d)
    ekf_rmse_h = rmse(cols["ekf_heading_deg"], vis_h)
    enc_rmse_h = rmse(cols["enc_heading_deg"], vis_h)

    fig, (ax_d, ax_h) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    # To nen DO cho cac khoang KHONG CO vision (vision_fresh=0).
    no_vision_label = False
    for t0, t1 in stale_ranges:
        kw = dict(color="red", alpha=0.08, zorder=0)
        if not no_vision_label:
            kw["label"] = "KHONG vision (vision_fresh=0)"
            no_vision_label = True
        ax_d.axvspan(t0, t1, **kw)
        ax_h.axvspan(t0, t1, color="red", alpha=0.08, zorder=0)

    # --- d(t) ---
    ax_d.plot(t, cols["ekf_d"], color="#1f77b4", lw=1.8, label=f"EKF fused (RMSE vs vision={ekf_rmse_d:.3f}m)")
    ax_d.plot(t, vis_d, color="#2ca02c", lw=1.4, label="vision thuan (tham chieu)")
    ax_d.plot(t, cols["enc_d"], color="#d62728", lw=1.4, ls="--",
              label=f"encoder thuan (RMSE vs vision={enc_rmse_d:.3f}m)")
    ax_d.axhline(0.0, color="0.7", lw=0.8, zorder=0)
    ax_d.set_ylabel("d [m]  (+d = line ben phai xe)")
    ax_d.set_title("So sanh lech ngang d — doan thang")
    ax_d.grid(True, alpha=0.3)
    ax_d.legend(loc="best", fontsize=9)

    # --- heading(t) ---
    ax_h.plot(t, cols["ekf_heading_deg"], color="#1f77b4", lw=1.8, label=f"EKF fused (RMSE vs vision={ekf_rmse_h:.2f}deg)")
    ax_h.plot(t, vis_h, color="#2ca02c", lw=1.4, label="vision thuan (tham chieu)")
    ax_h.plot(t, cols["enc_heading_deg"], color="#d62728", lw=1.4, ls="--",
              label=f"encoder thuan (RMSE vs vision={enc_rmse_h:.2f}deg)")
    ax_h.axhline(0.0, color="0.7", lw=0.8, zorder=0)
    ax_h.set_ylabel("heading [deg]")
    ax_h.set_xlabel("thoi gian [s]")
    ax_h.set_title("So sanh heading — doan thang")
    ax_h.grid(True, alpha=0.3)
    ax_h.legend(loc="best", fontsize=9)

    fig.tight_layout()

    n_stale = len(stale_ranges)
    stale_total = sum(t1 - t0 for t0, t1 in stale_ranges)
    print(f"Doc {len(t)} dong tu {args.csv}")
    print(f"  d  RMSE vs vision:  EKF={ekf_rmse_d:.4f} m   | encoder={enc_rmse_d:.4f} m")
    print(f"  hd RMSE vs vision:  EKF={ekf_rmse_h:.4f} deg | encoder={enc_rmse_h:.4f} deg")
    print(f"  Mat vision: {n_stale} khoang, tong {stale_total:.1f}s / {t[-1]:.1f}s "
          f"({100*stale_total/t[-1]:.1f}% thoi gian)")

    if args.out:
        fig.savefig(args.out, dpi=130)
        print(f"Da luu do thi -> {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
