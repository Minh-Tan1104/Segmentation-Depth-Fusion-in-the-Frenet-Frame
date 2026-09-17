#!/usr/bin/env python3
"""Vẽ file CSV quỹ đạo (do gps_node ghi khi log_latlon_enable) trong FRAME MÉT
CỤC BỘ — KHÔNG tải tile bản đồ (chạy offline, không cần mạng).

CSV có 16 cột (xem Gps/node.py): t_s, lat, lon, in_curve_zone, s_m, d_m,
sigma_d, raw_lat, raw_lon, raw_fix_type, raw_h_acc_m, enc_s_m, enc_d_m,
enc_sigma_d, enc_lat, enc_lon. Script vẽ 2 panel:
  1. SO SÁNH ĐỊNH VỊ (frame mét, trục X/Y tính bằng mét, lưới, tỉ lệ 1:1):
     Ground truth (--gt) + GPS thô (raw_lat/raw_lon) + RouteEKF (lat/lon,
     đã fusion) + encoder thuần dead-reckoning (enc_lat/enc_lon). Đoạn trong
     curve zone tô đậm và khoanh vòng nét đứt. Ground truth bị CẮT PHẦN ĐẦU
     để bắt đầu đúng từ điểm START của run (khớp với phạm vi run đi qua).
  2. LỆCH NGANG CÓ DẤU so với Ground Truth, CHỈ trong đoạn cua: chiếu vuông
     góc từng track (RouteEKF / GPS thô / encoder) lên polyline GT, lấy dấu
     theo quy ước + = lệch TRÁI GT, − = lệch PHẢI GT. So trực tiếp ai bám GT
     sát nhất qua cua. Không có --gt thì fallback về đồ thị d-vs-tuyến cũ.

Sai số vị trí (Euclid trong frame mét) của RouteEKF / GPS thô / encoder so với
ground truth được tính bằng phép chiếu điểm lên polyline GT, tách RIÊNG cho
đoạn CUA (in_curve_zone=1) — in ra console và annotate trên panel trái.

Thuần Python, KHÔNG rclpy, KHÔNG contextily. Cần: matplotlib, pyproj, numpy.
    pip install matplotlib pyproj

Dùng:
    python3 plot_run_map.py ~/rl_car_run_XXXX.csv
    python3 plot_run_map.py run.csv --gt map/gps_log.csv        # phủ + tính sai số vs GT
    python3 plot_run_map.py run.csv --curve-only                # chỉ vẽ đoạn in_curve_zone=1
    python3 plot_run_map.py run.csv --gt map/gps_log.csv --out map.png
"""
from __future__ import annotations

import argparse
import csv
import sys

import numpy as np

try:
    from pyproj import Transformer
except ModuleNotFoundError:
    sys.exit("Thieu pyproj. Cai: pip install pyproj")

WGS84 = "EPSG:4326"
LAT_KEYS = ("lat", "latitude", "y")
LON_KEYS = ("lon", "lng", "long", "longitude", "x")


def _find_col(fieldnames, keys):
    low = {f.lower().strip(): f for f in fieldnames}
    for k in keys:
        if k in low:
            return low[k]
    return None


def _floats(rows, col):
    out = []
    for r in rows:
        v = r.get(col, "")
        try:
            out.append(float(v))
        except (ValueError, TypeError):
            out.append(float("nan"))
    return np.array(out)


def load_run(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        sys.exit(f"CSV rong: {path}")
    return rows


def make_local_transformer(lat0, lon0):
    """Azimuthal equidistant đặt gốc tại (lat0, lon0) -> (x, y) mét cục bộ.
    CÙNG 1 transformer cho run + GPS + encoder + GT để mọi track chung frame."""
    return Transformer.from_crs(
        WGS84, f"+proj=aeqd +lat_0={lat0} +lon_0={lon0} +units=m", always_xy=True
    )


def to_xy(tf, lat, lon):
    x, y = tf.transform(np.asarray(lon), np.asarray(lat))
    return np.asarray(x), np.asarray(y)


def point_to_polyline(px, py, gx, gy, chunk=256):
    """Chiếu mỗi điểm (px,py) vuông góc lên polyline GT (gx,gy), lấy đoạn gần
    nhất. Chia khối theo điểm để giới hạn bộ nhớ. Trả 2 mảng (Np,):
      - dist   : khoảng cách Euclid [m] (không dấu, để tính RMSE/max).
      - signed : lệch ngang CÓ DẤU [m] theo quy ước +=TRÁI GT, −=PHẢI GT
        (dấu = tích chéo tiếp tuyến GT × vector lệch: t×o = tx*oy − ty*ox,
        dương khi điểm nằm bên trái chiều đi của GT).
    NaN nếu điểm không hợp lệ."""
    A = np.column_stack([gx[:-1], gy[:-1]])           # (Ns,2) đầu đoạn
    seg = np.column_stack([gx[1:] - gx[:-1], gy[1:] - gy[:-1]])  # (Ns,2)
    seglen2 = np.maximum((seg ** 2).sum(1), 1e-12)    # (Ns,)
    seglen = np.sqrt(seglen2)
    tang = seg / seglen[:, None]                      # (Ns,2) tiếp tuyến đơn vị
    out_d = np.full(len(px), np.nan)
    out_s = np.full(len(px), np.nan)
    for i in range(0, len(px), chunk):
        P = np.column_stack([px[i:i + chunk], py[i:i + chunk]])  # (m,2)
        ok = np.isfinite(P).all(1)
        if not ok.any():
            continue
        rel = P[:, None, :] - A[None, :, :]           # (m,Ns,2)
        t = np.clip((rel * seg[None]).sum(2) / seglen2[None], 0.0, 1.0)  # (m,Ns)
        closest = A[None] + t[..., None] * seg[None]  # (m,Ns,2)
        off = P[:, None, :] - closest                 # (m,Ns,2)
        dist = np.linalg.norm(off, axis=2)            # (m,Ns)
        j = np.argmin(dist, axis=1)                   # (m,) đoạn gần nhất
        rng = np.arange(P.shape[0])
        best_d = dist[rng, j]
        ox = off[rng, j, 0]; oy = off[rng, j, 1]
        tx = tang[j, 0]; ty = tang[j, 1]
        signed = tx * oy - ty * ox                    # + = trái GT
        best_d[~ok] = np.nan
        signed[~ok] = np.nan
        out_d[i:i + chunk] = best_d
        out_s[i:i + chunk] = signed
    return out_d, out_s


def contiguous_segments(mask):
    """Trả danh sách (i0, i1) các đoạn True liên tiếp trong mask (i1 exclusive)."""
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(idx) > 1)
    starts = np.concatenate([[0], breaks + 1])
    ends = np.concatenate([breaks + 1, [idx.size]])
    return [(int(idx[s]), int(idx[e - 1]) + 1) for s, e in zip(starts, ends)]


def err_stats(dist, mask=None):
    d = dist if mask is None else dist[mask]
    d = d[np.isfinite(d)]
    if d.size == 0:
        return None
    return dict(n=d.size, mean=float(d.mean()), rmse=float(np.sqrt((d ** 2).mean())),
                med=float(np.median(d)), p95=float(np.percentile(d, 95)), max=float(d.max()))


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Ve CSV quy dao trong frame met cuc bo (khong tai tile ban do)."
    )
    ap.add_argument("run_csv", help="CSV do gps_node ghi (log_latlon_enable).")
    ap.add_argument("--gt", default=None,
                    help="CSV ground truth (lat/lon). Ve + TINH SAI SO vi tri (tach rieng doan cua).")
    ap.add_argument("--curve-only", action="store_true", help="Chi ve diem in_curve_zone=1.")
    ap.add_argument("--no-gt-trim", action="store_true",
                    help="Ve toan bo ground truth (mac dinh: cat phan dau, bat dau tu START cua run).")
    ap.add_argument("--out", default=None, help="Luu PNG thay vi mo cua so.")
    ap.add_argument("--zoom", type=int, default=None, help="(bo qua — khong con tai tile ban do).")
    args = ap.parse_args()

    rows = load_run(args.run_csv)
    fld = rows[0].keys()
    lat = _floats(rows, _find_col(fld, LAT_KEYS))
    lon = _floats(rows, _find_col(fld, LON_KEYS))
    in_zone = _floats(rows, "in_curve_zone") if "in_curve_zone" in fld else np.zeros(len(rows))
    has_raw = "raw_lat" in fld and "raw_lon" in fld
    r_lat = _floats(rows, "raw_lat") if has_raw else np.full(len(rows), np.nan)
    r_lon = _floats(rows, "raw_lon") if has_raw else np.full(len(rows), np.nan)
    has_enc_latlon = "enc_lat" in fld and "enc_lon" in fld
    e_lat = _floats(rows, "enc_lat") if has_enc_latlon else np.full(len(rows), np.nan)
    e_lon = _floats(rows, "enc_lon") if has_enc_latlon else np.full(len(rows), np.nan)
    s_m = _floats(rows, "s_m") if "s_m" in fld else None
    d_m = _floats(rows, "d_m") if "d_m" in fld else None
    enc_s = _floats(rows, "enc_s_m") if "enc_s_m" in fld else None
    enc_d = _floats(rows, "enc_d_m") if "enc_d_m" in fld else None

    curve_mask = in_zone >= 0.5
    map_sel = np.isfinite(lat) & np.isfinite(lon)
    if args.curve_only:
        map_sel &= curve_mask
        print(f"--curve-only: keeping {int(map_sel.sum())}/{len(map_sel)} points")
    if map_sel.sum() < 1:
        sys.exit("Khong co diem lat/lon hop le (hoac khong co diem trong cua voi --curve-only).")

    # --- Frame mét cục bộ: gốc tại tâm quỹ đạo run, dùng chung cho mọi track ---
    lat0 = float(np.nanmean(lat[map_sel]))
    lon0 = float(np.nanmean(lon[map_sel]))
    tf = make_local_transformer(lat0, lon0)
    ex, ey = to_xy(tf, lat, lon)                  # RouteEKF (full length, NaN chỗ thiếu)
    zone_ok = curve_mask & map_sel

    try:
        import matplotlib
        if args.out:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle
    except ModuleNotFoundError:
        sys.exit("Thieu matplotlib. Cai: pip install matplotlib")

    have_ds = s_m is not None and d_m is not None
    ncols = 2 if have_ds else 1
    fig, axes = plt.subplots(1, ncols, figsize=(8 * ncols, 7))
    ax_map = axes[0] if have_ds else axes

    # --- Ground truth: chiếu + CẮT PHẦN ĐẦU cho khớp phạm vi run ---
    gx = gy = None
    if args.gt:
        gt_rows = load_run(args.gt)
        gfld = gt_rows[0].keys()
        g_lat = _floats(gt_rows, _find_col(gfld, LAT_KEYS))
        g_lon = _floats(gt_rows, _find_col(gfld, LON_KEYS))
        gok = np.isfinite(g_lat) & np.isfinite(g_lon)
        gx_full, gy_full = to_xy(tf, g_lat[gok], g_lon[gok])

        gx, gy = gx_full, gy_full  # bản đầy đủ dùng để TÍNH SAI SỐ (chiếu chuẩn)
        if args.no_gt_trim:
            gx_plot, gy_plot = gx_full, gy_full
        else:
            # Cắt: tìm đỉnh GT gần điểm START và END của run, chỉ vẽ lát giữa
            # -> ground truth bắt đầu đúng từ start, không thừa đầu tuyến.
            run_i = np.flatnonzero(map_sel)
            sx, sy = ex[run_i[0]], ey[run_i[0]]
            fx, fy = ex[run_i[-1]], ey[run_i[-1]]
            i0 = int(np.argmin((gx_full - sx) ** 2 + (gy_full - sy) ** 2))
            i1 = int(np.argmin((gx_full - fx) ** 2 + (gy_full - fy) ** 2))
            lo, hi = (i0, i1) if i0 <= i1 else (i1, i0)
            gx_plot, gy_plot = gx_full[lo:hi + 1], gy_full[lo:hi + 1]
            print(f"Ground truth: trimmed head, drawing vertices {lo}..{hi} "
                  f"({hi - lo + 1}/{gx_full.size} points, starting at run START)")
        ax_map.plot(gx_plot, gy_plot, "--", color="#2ca02c", lw=2.0, zorder=3,
                    label="Ground truth (RTK)")

    # GNSS thô: chấm mờ dưới cùng.
    raw_sel = np.isfinite(r_lat) & np.isfinite(r_lon)
    if args.curve_only:
        raw_sel &= curve_mask
    if raw_sel.any():
        rx, ry = to_xy(tf, r_lat[raw_sel], r_lon[raw_sel])
        ax_map.scatter(rx, ry, c="#1f77b4", s=4, alpha=0.35, zorder=4, label="GNSS raw")

    # Encoder thuần dead-reckoning (không anchor).
    enc_sel = np.isfinite(e_lat) & np.isfinite(e_lon)
    if args.curve_only:
        enc_sel &= curve_mask
    if enc_sel.any():
        qx, qy = to_xy(tf, e_lat[enc_sel], e_lon[enc_sel])
        ax_map.plot(qx, qy, "-", color="#ff7f0e", lw=1.3, alpha=0.85, zorder=5,
                    label="Encoder DR, unanchored")

    # RouteEKF: đường chính, ngoài cua đỏ mảnh (fused), trong cua đỏ dày
    # (anchored dead-reckon — GPS/vision bị chặn, xem Gps/route_ekf.py).
    plot_sel = map_sel & curve_mask if args.curve_only else map_sel
    ax_map.plot(ex[plot_sel], ey[plot_sel], "-", color="#d62728", lw=1.6, zorder=6,
                label="Route-frame (fused, outside zone)")
    if zone_ok.any():
        # Vẽ từng đoạn cua liền mạch để không nối ngang giữa các zone.
        first = True
        for i0, i1 in contiguous_segments(zone_ok):
            ax_map.plot(ex[i0:i1], ey[i0:i1], "-", color="#8b0000", lw=3.2, zorder=7,
                        label="Route-frame (anchored DR, in zone)" if first else None)
            first = False

    run_i = np.flatnonzero(map_sel)
    ax_map.scatter(ex[run_i[0]], ey[run_i[0]], c="green", s=90, marker="o",
                   edgecolors="k", zorder=9, label="start")
    ax_map.scatter(ex[run_i[-1]], ey[run_i[-1]], c="red", s=90, marker="s",
                   edgecolors="k", zorder=9, label="end")

    # Khoanh vòng nét đứt quanh từng đoạn cua (highlight như hình mẫu).
    circ_label = True
    for i0, i1 in contiguous_segments(zone_ok):
        cx, cy = np.nanmean(ex[i0:i1]), np.nanmean(ey[i0:i1])
        rad = np.nanmax(np.hypot(ex[i0:i1] - cx, ey[i0:i1] - cy)) + 3.0
        ax_map.add_patch(Circle((cx, cy), rad, fill=False, ls="--", lw=1.8,
                                edgecolor="#17becf", zorder=8,
                                label="Curve zone" if circ_label else None))
        circ_label = False

    # --- Sai số vị trí so với ground truth, tách riêng đoạn cua ---
    # signed_by_name: lệch ngang CÓ DẤU (+trái GT / −phải GT) của từng track,
    # giữ lại để vẽ panel phải (đoạn cua). Cùng phép chiếu với sai số RMSE.
    err_text = None
    signed_by_name = {}
    if gx is not None:
        print("\nPOSITION ERROR vs ground truth [m] (perpendicular projection onto GT polyline):")
        summary_lines = []
        for name, (la, lo_) in (("RouteEKF", (lat, lon)),
                                ("GPS raw ", (r_lat, r_lon)),
                                ("Encoder ", (e_lat, e_lon))):
            px, py = to_xy(tf, la, lo_)
            dist, signed = point_to_polyline(px, py, gx, gy)
            signed_by_name[name.strip()] = signed
            allst = err_stats(dist)
            cvst = err_stats(dist, curve_mask)
            stst = err_stats(dist, ~curve_mask)
            if allst is None:
                continue
            print(f"  {name}: OVERALL    mean={allst['mean']:.3f} rmse={allst['rmse']:.3f} "
                  f"med={allst['med']:.3f} max={allst['max']:.3f}  (n={allst['n']})")
            if cvst:
                print(f"           IN CURVE   mean={cvst['mean']:.3f} rmse={cvst['rmse']:.3f} "
                      f"med={cvst['med']:.3f} max={cvst['max']:.3f}  (n={cvst['n']})")
            if stst:
                print(f"           OFF CURVE  mean={stst['mean']:.3f} rmse={stst['rmse']:.3f} "
                      f"med={stst['med']:.3f} max={stst['max']:.3f}  (n={stst['n']})")
            if name.strip() == "RouteEKF":
                summary_lines.append("RouteEKF vs GT [m]:")
                summary_lines.append(f"  overall: RMSE {allst['rmse']:.3f}  max {allst['max']:.2f}")
                if cvst:
                    summary_lines.append(f"  in curve : RMSE {cvst['rmse']:.3f}  max {cvst['max']:.2f}")
                if stst:
                    summary_lines.append(f"  off curve: RMSE {stst['rmse']:.3f}  max {stst['max']:.2f}")
        err_text = "\n".join(summary_lines)

    # Sai số RMSE (kể cả tách riêng đoạn cua) đã in đầy đủ ra console ở trên —
    # KHÔNG phủ hộp text lên map để khỏi che quỹ đạo khi zoom.
    ax_map.set_aspect("equal", adjustable="datalim")
    ax_map.set_xlabel("X (m)")
    ax_map.set_ylabel("Y (m)")
    ax_map.grid(True, alpha=0.3)
    ax_map.set_title("Localization comparison")
    # Legend "upper right" hay đè lên chính quỹ đạo (vd điểm "end" nằm ở đỉnh
    # trục Y như run này) — không phải lỗi icon, mà legend che dữ liệu thật.
    # Dùng margins() (KHÔNG tự set_ylim bằng tay) để nới chỗ: set_aspect
    # "equal, datalim" tính lại giới hạn trục LÚC VẼ dựa trên toàn bộ datalim
    # (gồm cả patch Circle của curve zone) — set_ylim thủ công trước đó bị nó
    # ghi đè và cắt cụt hình tròn vì không biết bán kính patch. margins() cộng
    # lề vào chính datalim nên luôn bao trọn mọi artist, kể cả Circle.
    ax_map.margins(y=0.38)
    map_legend = ax_map.legend(loc="upper right", fontsize=8)
    # scatter() truyền y nguyên 'x' (kích thước điểm THẬT trên map) vào legend
    # -> "GNSS raw" (s=4, gần như vô hình) và "end" (s=90) lệch quá xa nhau,
    # icon to đè lên/che dòng chữ kế bên. Ép mọi marker trong legend về 1 cỡ
    # thống nhất — không đụng tới kích thước điểm thật trên bản đồ.
    # legend_handles (snake_case) chỉ có tu matplotlib>=3.7; ban cu (vd 3.5)
    # chi co legendHandles (camelCase, da deprecated o ban moi) -> fallback.
    legend_handles = getattr(map_legend, "legend_handles", None)
    if legend_handles is None:
        legend_handles = map_legend.legendHandles
    for handle in legend_handles:
        if hasattr(handle, "set_sizes"):
            handle.set_sizes([36])

    # --- Panel phải: lệch ngang CÓ DẤU so với GT, CHỈ trong đoạn cua ---
    # Quy ước dấu: + = xe lệch sang TRÁI của ground truth, − = lệch sang PHẢI.
    # Vẽ 3 track (RouteEKF / GPS thô / encoder thuần) theo s để so trực tiếp
    # ai bám GT sát nhất qua cua. Cần --gt (không có GT thì fallback d-vs-tuyến).
    if have_ds:
        ax_ds = axes[1]
        if signed_by_name and curve_mask.any():
            cz_label = True
            for i0, i1 in contiguous_segments(curve_mask):
                ax_ds.axvspan(np.nanmin(s_m[i0:i1]), np.nanmax(s_m[i0:i1]),
                              color="#17becf", alpha=0.12, zorder=0,
                              label="Curve zone" if cz_label else None)
                cz_label = False
            # Chỉ giữ mẫu trong cua; NaN ngoài cua để đường ngắt đúng ranh giới.
            s_cv = np.where(curve_mask, s_m, np.nan)
            order = np.argsort(np.where(np.isfinite(s_cv), s_cv, np.inf))
            for name, color, ls, lbl in (("RouteEKF", "#d62728", "-", "Route-frame, anchored DR"),
                                         ("GPS raw", "#1f77b4", ":", "GNSS raw"),
                                         ("Encoder", "#ff7f0e", "--", "Encoder DR, unanchored")):
                sig = signed_by_name.get(name)
                if sig is None:
                    continue
                y = np.where(curve_mask, sig, np.nan)
                ax_ds.plot(s_cv[order], y[order], color=color, ls=ls, lw=1.7, label=lbl)
            ax_ds.axhline(0.0, color="0.5", lw=1.0)
            ax_ds.set_xlabel("s (m)")
            ax_ds.set_ylabel("lateral deviation (m)")
            ax_ds.set_title("Lateral deviation from ground truth (RTK), in curve")
            # Giới hạn trục x quanh vùng cua cho dễ đọc (+ biên 5 m).
            s_in = s_m[curve_mask & np.isfinite(s_m)]
            if s_in.size:
                ax_ds.set_xlim(s_in.min() - 5.0, s_in.max() + 5.0)
        else:
            # Fallback khi không có --gt: giữ đồ thị d-vs-tuyến như cũ.
            ds_mask = curve_mask if args.curve_only else np.ones(len(s_m), bool)
            for i0, i1 in contiguous_segments(curve_mask):
                ax_ds.axvspan(np.nanmin(s_m[i0:i1]), np.nanmax(s_m[i0:i1]),
                              color="#17becf", alpha=0.12, zorder=0)
            ax_ds.plot(s_m[ds_mask], d_m[ds_mask], color="#d62728", lw=1.6,
                       label="RouteEKF d (fused)")
            if enc_s is not None and enc_d is not None:
                ax_ds.plot(enc_s[ds_mask], enc_d[ds_mask], color="#1f77b4", lw=1.4, ls="--",
                           label="encoder-only d (no GPS/vision)")
            ax_ds.axhline(0.0, color="0.7", lw=0.8)
            ax_ds.set_xlabel("s (m)")
            ax_ds.set_ylabel("d (m)  (+R)")
            ax_ds.set_title("Lateral offset (no --gt)")
        ax_ds.grid(True, alpha=0.3)
        ax_ds.legend(loc="upper right", fontsize=9)

    fig.tight_layout()
    print(f"\nRead {len(rows)} rows from {args.run_csv} "
          f"({int(zone_ok.sum())} points in curve)")
    if args.out:
        fig.savefig(args.out, dpi=130)
        print(f"Saved -> {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
