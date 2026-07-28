#!/usr/bin/env python3
"""Thống kê + vẽ TIMING toàn pipeline từ CSV do control_node + perception_node ghi.

Nhận 1 HOẶC NHIỀU file CSV (gộp lại theo cột 'loop'):
  - control_node (rl_car_timing_*.csv): loop = 'control' (~50 Hz), 'planner' (~15 Hz).
    Cột: t_s, loop, period_ms, exec_ms.
  - perception_node (rl_car_perc_timing_*.csv): loop = 'perception' (nhịp camera).
    Cột: t_s, loop, period_ms, exec_ms, seg_ms, det_ms, frame_age_ms.

period_ms = chu kỳ THẬT giữa 2 lần gọi cùng loop (không throttle) -> Hz hiệu dụng.
exec_ms   = thời gian chạy callback. seg_ms/det_ms = 2 lần YOLO (chỉ perception).
frame_age_ms = now − camera stamp lúc bắt đầu xử lý = độ trễ frame (chỉ perception).

In BẢNG (Hz hiệu dụng, mean/p50/p95/p99/max period, jitter σ, exec p95, và với
perception thêm seg/det/age p95) rồi vẽ. Loại warmup + gap lớn (pause/stall).

Dùng:
  python3 scripts/plot_loop_timing.py ~/rl_car_timing_*.csv
  python3 scripts/plot_loop_timing.py ctrl.csv perc.csv --warmup 10 --out timing.png
  python3 scripts/plot_loop_timing.py <csv...> --warmup 10 --max-gap-ms 500
"""
from __future__ import annotations

import argparse
import csv
import math
import statistics as st
import sys

# Cột phụ (chỉ perception có) — đọc nếu tồn tại, bỏ qua nếu không.
EXTRA = ("seg_ms", "det_ms", "frame_age_ms")
LOOP_ORDER = ["perception", "control", "planner"]


def load(paths):
    rows = {}
    for path in paths:
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                loop = r.get("loop", "")
                if not loop:
                    continue
                d = rows.setdefault(loop, {"t": [], "period": [], "exec": [],
                                           "seg_ms": [], "det_ms": [], "frame_age_ms": []})
                def fl(k):
                    v = r.get(k, "")
                    try:
                        return float(v)
                    except (ValueError, TypeError):
                        return float("nan")
                d["t"].append(fl("t_s"))
                d["period"].append(fl("period_ms"))
                d["exec"].append(fl("exec_ms"))
                for k in EXTRA:
                    d[k].append(fl(k))
    return rows


def clean(vals, ts, warmup_s, max_gap_ms, gate=None):
    """Bỏ NaN, warmup đầu. gate (nếu có) = mảng period để loại pause/stall theo
    cùng chỉ số (dùng khi lọc seg/det/age theo period của chính frame đó)."""
    out = []
    for i, (v, t) in enumerate(zip(vals, ts)):
        if math.isnan(v) or t < warmup_s:
            continue
        g = gate[i] if gate is not None else v
        if max_gap_ms and not math.isnan(g) and g > max_gap_ms:
            continue
        out.append(v)
    return out


def pct(d, q):
    d = sorted(d)
    return d[min(len(d) - 1, int(q * len(d)))]


def main():
    ap = argparse.ArgumentParser(description="Thong ke + ve timing pipeline (control + perception).")
    ap.add_argument("csv", nargs="+", help="1+ CSV log_timing (control_node va/hoac perception_node).")
    ap.add_argument("--out", default=None, help="Luu PNG thay vi mo cua so.")
    ap.add_argument("--warmup", type=float, default=3.0, help="Bo N giay dau (mac dinh 3).")
    ap.add_argument("--max-gap-ms", type=float, default=500.0,
                    help="Bo period > nguong nay (pause/stall; mac dinh 500ms). 0 = giu het.")
    args = ap.parse_args()

    data = load(args.csv)
    order = [k for k in LOOP_ORDER if k in data and data[k]["t"]]
    order += [k for k in data if k not in LOOP_ORDER and data[k]["t"]]
    if not order:
        sys.exit("CSV khong co dong nao — dung file log_timing_enable khong?")

    # --- Bảng ---
    print(f"\nPIPELINE TIMING — {', '.join(args.csv)}")
    print(f"(warmup <{args.warmup}s + period >{args.max_gap_ms:.0f}ms excluded)\n")
    hdr = (f"{'loop':11s}|{'n':>6}|{'eff Hz':>7}|{'mean':>7}|{'p50':>7}|{'p95':>7}|"
           f"{'p99':>7}|{'max':>7}|{'jit σ':>7}|{'exec p95':>9}|{'seg p95':>8}|"
           f"{'det p95':>8}|{'age p95':>8}")
    print(hdr)
    print(f"{'':11s}|{'':>6}|{'':>7}|{'ms':>7}|{'ms':>7}|{'ms':>7}|{'ms':>7}|"
          f"{'ms':>7}|{'ms':>7}|{'ms':>9}|{'ms':>8}|{'ms':>8}|{'ms':>8}")
    print("-" * len(hdr))
    clean_by_loop = {}
    for name in order:
        d = data[name]
        per = clean(d["period"], d["t"], args.warmup, args.max_gap_ms)
        ex = clean(d["exec"], d["t"], args.warmup, args.max_gap_ms, gate=d["period"])
        seg = clean(d["seg_ms"], d["t"], args.warmup, args.max_gap_ms, gate=d["period"])
        det = clean(d["det_ms"], d["t"], args.warmup, args.max_gap_ms, gate=d["period"])
        age = clean(d["frame_age_ms"], d["t"], args.warmup, args.max_gap_ms, gate=d["period"])
        clean_by_loop[name] = dict(per=per, ex=ex, seg=seg, det=det, age=age, raw=d)
        if not per:
            print(f"{name:11s}| (khong du mau sau khi loc)")
            continue
        mean = sum(per) / len(per)
        segc = f"{pct(seg,.95):8.2f}" if seg else f"{'—':>8}"
        detc = f"{pct(det,.95):8.2f}" if det else f"{'—':>8}"
        agec = f"{pct(age,.95):8.2f}" if age else f"{'—':>8}"
        print(f"{name:11s}|{len(per):6d}|{1000/mean:7.2f}|{mean:7.2f}|{pct(per,.50):7.2f}|"
              f"{pct(per,.95):7.2f}|{pct(per,.99):7.2f}|{max(per):7.2f}|"
              f"{st.pstdev(per):7.3f}|{pct(ex,.95):9.3f}|{segc}|{detc}|{agec}")
    print("-" * len(hdr))
    print("eff Hz=1000/mean(period). jit σ=std(period). exec=callback runtime.")
    print("seg/det=2 lan YOLO, age=frame delay (now−camera stamp) — chi perception.")

    # --- Vẽ ---
    try:
        import matplotlib
        if args.out:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        if args.out:
            sys.exit("Thieu matplotlib. Cai: pip install matplotlib")
        return

    color = {"control": "#d62728", "planner": "#1f77b4", "perception": "#2ca02c"}
    fig, (ax_h, ax_e) = plt.subplots(2, 1, figsize=(11, 7))

    for name in order:
        cb = clean_by_loop[name]
        if not cb["per"]:
            continue
        c = color.get(name, "#7f7f7f")
        mean = sum(cb["per"]) / len(cb["per"])
        ax_h.hist(cb["per"], bins=60, color=c, alpha=0.5,
                  label=f"{name} ({1000/mean:.1f} Hz, σ={st.pstdev(cb['per']):.2f} ms)")
        ax_h.axvline(mean, color=c, ls="--", lw=1.2)
    ax_h.set_xlabel("loop period (ms)")
    ax_h.set_ylabel("count")
    ax_h.set_title("Loop period distribution (real interval)")
    ax_h.grid(True, alpha=0.3)
    ax_h.legend(loc="upper right", fontsize=9)

    for name in order:
        d = clean_by_loop[name]["raw"]
        c = color.get(name, "#7f7f7f")
        tt = [t for t, v in zip(d["t"], d["exec"]) if t >= args.warmup and not math.isnan(v)]
        ee = [v for t, v in zip(d["t"], d["exec"]) if t >= args.warmup and not math.isnan(v)]
        ax_e.plot(tt, ee, color=c, lw=0.8, alpha=0.8, label=f"{name} exec")
    ax_e.set_xlabel("time (s)")
    ax_e.set_ylabel("callback exec (ms)")
    ax_e.set_title("Per-tick execution time")
    ax_e.grid(True, alpha=0.3)
    ax_e.legend(loc="upper right", fontsize=9)

    fig.suptitle("Pipeline timing", fontsize=13, fontweight="bold")
    fig.tight_layout()
    if args.out:
        fig.savefig(args.out, dpi=130)
        print(f"\nSaved plot -> {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
