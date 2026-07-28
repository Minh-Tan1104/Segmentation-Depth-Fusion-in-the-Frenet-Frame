#!/usr/bin/env python3
"""Vẽ đồ thị so sánh (d, heading) 3 nguồn từ CSV do control_node ghi.

CSV được control_node ghi khi bật log_compare_enable (xem
config/rl_car_params.yaml + control/node.py:_maybe_log_compare). Chạy closed
loop (đầu tuyến né vật cản, phần sau bám làn), mỗi dòng là 1 thời điểm ở chế
độ không-curve, 3 cặp (d, heading) cùng quy ước vision (d = d_meters, +d =
line/lệch sang phải; heading = độ):
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


def mae(a: list[float], b: list[float]) -> float:
    """MAE giữa 2 chuỗi, bỏ qua cặp có NaN."""
    ae = [abs(x - y) for x, y in zip(a, b) if not (math.isnan(x) or math.isnan(y))]
    return sum(ae) / len(ae) if ae else float("nan")


def max_err(a: list[float], b: list[float]) -> float:
    """Sai số lớn nhất (abs) giữa 2 chuỗi, bỏ qua cặp có NaN."""
    ae = [abs(x - y) for x, y in zip(a, b) if not (math.isnan(x) or math.isnan(y))]
    return max(ae) if ae else float("nan")


def _errstats(a: list[float], b: list[float], idx: list[int]) -> tuple[float, float, float]:
    """(RMSE, MAE, MAX) giữa a[idx] và b[idx]."""
    sub_a = [a[i] for i in idx]
    sub_b = [b[i] for i in idx]
    return rmse(sub_a, sub_b), mae(sub_a, sub_b), max_err(sub_a, sub_b)


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
    # Ban goc, KHONG mask boi --mask-stale-vision — dung rieng de tinh RMSE/MAE/MAX
    # (vision_d/vision_heading_deg da duoc control_node ghi "dong lai" gia tri cu
    # kem vision_fresh=0 khi mat vision, xem control/node.py:_maybe_log_compare).
    vis_d_raw = list(cols["vision_d"])
    vis_h_raw = list(cols["vision_heading_deg"])

    # Phat hien khoang mat vision (dung de to nen + tach thong ke with/no vision).
    has_fresh_col = "vision_fresh" in cols
    if has_fresh_col:
        fresh_mask = cols["vision_fresh"]
    else:
        # Fallback: vision mat = NaN trong vision_d.
        fresh_mask = [0.0 if math.isnan(d) else 1.0 for d in vis_d]
    stale_ranges = _find_stale_ranges(t, fresh_mask)

    if args.mask_stale_vision and has_fresh_col:
        vis_d = [d if fr >= 0.5 else float("nan") for d, fr in zip(vis_d, cols["vision_fresh"])]
        vis_h = [h if fr >= 0.5 else float("nan") for h, fr in zip(vis_h, cols["vision_fresh"])]

    # Vision là nguồn RỜI RẠC: chỉ có giá trị mới khi camera detect được frame
    # (giữa các frame CSV lặp lại giá trị cũ). Lọc lấy đúng các mẫu vision "mới"
    # (đổi giá trị so với mẫu trước, không NaN) để vẽ bằng CHẤM RỜI — thể hiện
    # đúng bản chất rời rạc, tương phản với encoder/EKF chạy liên tục mỗi tick.
    def _fresh_points(t_all, v_all):
        tp, vp = [], []
        prev = None
        for ti, vi in zip(t_all, v_all):
            if not math.isnan(vi) and vi != prev:
                tp.append(ti); vp.append(vi); prev = vi
        return tp, vp

    vis_d_t, vis_d_pts = _fresh_points(t, vis_d)
    vis_h_t, vis_h_pts = _fresh_points(t, vis_h)

    # Sai số so với vision (vision coi như tham chiếu đo trực tiếp làn), tách
    # riêng đoạn CÓ vision (fresh) và đoạn MẤT vision (stale, so với giá trị
    # vision "đông lại" cuối cùng — đúng mục đích của script: đo encoder/EKF
    # trôi bao nhiêu so với lần thấy làn gần nhất).
    fresh_idx = [i for i, fr in enumerate(fresh_mask) if fr >= 0.5]
    stale_idx = [i for i, fr in enumerate(fresh_mask) if fr < 0.5]

    ekf_d_with = _errstats(cols["ekf_d"], vis_d_raw, fresh_idx)
    enc_d_with = _errstats(cols["enc_d"], vis_d_raw, fresh_idx)
    ekf_d_no = _errstats(cols["ekf_d"], vis_d_raw, stale_idx)
    enc_d_no = _errstats(cols["enc_d"], vis_d_raw, stale_idx)
    ekf_h_with = _errstats(cols["ekf_heading_deg"], vis_h_raw, fresh_idx)
    enc_h_with = _errstats(cols["enc_heading_deg"], vis_h_raw, fresh_idx)
    ekf_h_no = _errstats(cols["ekf_heading_deg"], vis_h_raw, stale_idx)
    enc_h_no = _errstats(cols["enc_heading_deg"], vis_h_raw, stale_idx)

    fig, (ax_d, ax_h) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    # Tô nền các khoảng KHÔNG CÓ vision (vision_fresh=0) — cùng kiểu tô vùng
    # đặc biệt như "curve zone" ở panel phải run_map (dùng màu cyan nhạt).
    no_vision_label = False
    for t0, t1 in stale_ranges:
        kw = dict(color="#17becf", alpha=0.12, zorder=0)
        if not no_vision_label:
            kw["label"] = "no vision"
            no_vision_label = True
        ax_d.axvspan(t0, t1, **kw)
        ax_h.axvspan(t0, t1, color="#17becf", alpha=0.12, zorder=0)

    # Màu/kiểu đường theo vai trò, khớp panel phải run_map: ước lượng fusion =
    # đỏ liền, tham chiếu (vision, như Ground truth) = xanh lá đứt, encoder
    # thuần = cam đứt. Legend dùng "with vision" (không viết tắt).
    # Encoder + EKF = ĐƯỜNG LIỀN (chạy liên tục mỗi tick). Vision = CHẤM RỜI
    # (chỉ có tại các frame camera detect được) -> nhìn là thấy vision thưa
    # hơn hẳn, còn encoder/EKF dày đặc liên tục.
    # zorder: vision DƯỚI (2) để chấm không che đường; EKF TRÊN CÙNG (4) để
    # đường luôn thấy được kể cả đoạn đầu vision dày đặc quanh 0. Chấm vision
    # nhỏ + hơi trong (alpha) để không thành mảng đặc che mất đường bên dưới.
    # --- d(t) ---
    ax_d.plot(vis_d_t, vis_d_pts, color="#2ca02c", ls="none", marker="o", ms=3.5,
              alpha=0.7, mec="none", zorder=2, label="Vision")
    ax_d.plot(t, cols["enc_d"], color="#ff7f0e", lw=1.7, zorder=3, label="Encoder")
    ax_d.plot(t, cols["ekf_d"], color="#d62728", lw=1.7, zorder=4, label="EKF")
    ax_d.axhline(0.0, color="0.5", lw=1.0, zorder=0)
    ax_d.set_ylabel("d (m)")
    ax_d.set_title("Lateral offset with Vision")
    ax_d.grid(True, alpha=0.3)
    ax_d.legend(loc="upper left", fontsize=9)

    # --- heading(t) ---
    ax_h.plot(vis_h_t, vis_h_pts, color="#2ca02c", ls="none", marker="o", ms=3.5,
              alpha=0.7, mec="none", zorder=2, label="Vision")
    ax_h.plot(t, cols["enc_heading_deg"], color="#ff7f0e", lw=1.7, zorder=3, label="Encoder")
    ax_h.plot(t, cols["ekf_heading_deg"], color="#d62728", lw=1.7, zorder=4, label="EKF")
    ax_h.axhline(0.0, color="0.5", lw=1.0, zorder=0)
    ax_h.set_ylabel("heading (deg)")
    ax_h.set_xlabel("time (s)")
    ax_h.set_title("Heading with Vision")
    ax_h.grid(True, alpha=0.3)
    ax_h.legend(loc="upper left", fontsize=9)

    # Tiêu đề lớn cho cả hình: mô tả kịch bản chạy (closed loop, đầu né vật
    # cản rồi đi thẳng bám làn).
    fig.suptitle("Closed-loop run: straight-line lane keeping + obstacle avoidance",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()

    def _fmt(stats: tuple[float, float, float], unit: str) -> str:
        r, m, mx = stats
        return f"RMSE={r:.4f}{unit} MAE={m:.4f}{unit} MAX={mx:.4f}{unit}"

    n_stale = len(stale_ranges)
    stale_total = sum(t1 - t0 for t0, t1 in stale_ranges)
    print(f"Read {len(t)} rows from {args.csv}")
    print(f"  d  with vision: EKF {_fmt(ekf_d_with, ' m')} | encoder {_fmt(enc_d_with, ' m')}")
    print(f"  d  no vision  : EKF {_fmt(ekf_d_no, ' m')} | encoder {_fmt(enc_d_no, ' m')}")
    print(f"  hd with vision: EKF {_fmt(ekf_h_with, ' deg')} | encoder {_fmt(enc_h_with, ' deg')}")
    print(f"  hd no vision  : EKF {_fmt(ekf_h_no, ' deg')} | encoder {_fmt(enc_h_no, ' deg')}")
    print(f"  No vision: {n_stale} interval(s), total {stale_total:.1f}s / {t[-1]:.1f}s "
          f"({100*stale_total/t[-1]:.1f}% of time)")

    if args.out:
        fig.savefig(args.out, dpi=130)
        print(f"Saved plot -> {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
