#!/usr/bin/env python3
"""Vẽ file CSV quỹ đạo (do gps_node ghi khi log_latlon_enable) lên BẢN ĐỒ NỀN.

CSV có 16 cột (xem Gps/node.py): t_s, lat, lon, in_curve_zone, s_m, d_m,
sigma_d, raw_lat, raw_lon, raw_fix_type, raw_h_acc_m, enc_s_m, enc_d_m,
enc_sigma_d, enc_lat, enc_lon. Script vẽ:
  1. BẢN ĐỒ (matplotlib + contextily, tile OSM thật): quỹ đạo localization
     RouteEKF (lat/lon) + GPS thô (raw_lat/raw_lon) + encoder thuần
     dead-reckoning (enc_lat/enc_lon, suy từ (x,y) của route_ekf_enc — xem
     Gps/node.py) — đoạn trong cua tô đậm. Tuỳ chọn --gt phủ thêm ground
     truth RTK (chỉ vẽ, KHÔNG tính sai số). CSV cũ (14 cột, chưa có
     enc_lat/enc_lon) vẫn đọc được — chỉ bỏ qua track encoder trên bản đồ.
  2. Đồ thị d theo s (route frame): d_m (RouteEKF) vs enc_d_m (encoder thuần)
     — thấy encoder trôi bao nhiêu khi không có GPS/vision sửa.

Thuần Python, KHÔNG rclpy. Cần: matplotlib, contextily, pyproj (numpy).
    pip install matplotlib contextily pyproj

Dùng:
    python3 plot_run_map.py ~/rl_car_run_XXXX.csv
    python3 plot_run_map.py run.csv --gt rtk_vach.csv        # phủ thêm ground truth
    python3 plot_run_map.py run.csv --curve-only              # chỉ vẽ đoạn in_curve_zone=1
    python3 plot_run_map.py run.csv --out map.png --zoom 19   # lưu PNG, chỉnh zoom tile
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
WEB_MERCATOR = "EPSG:3857"
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


def to_mercator(lat, lon):
    tf = Transformer.from_crs(WGS84, WEB_MERCATOR, always_xy=True)
    x, y = tf.transform(lon, lat)
    return np.asarray(x), np.asarray(y)


def main() -> None:
    ap = argparse.ArgumentParser(description="Ve CSV quy dao len ban do nen (contextily).")
    ap.add_argument("run_csv", help="CSV do gps_node ghi (log_latlon_enable).")
    ap.add_argument("--gt", default=None, help="CSV ground truth RTK (chi ve, khong tinh sai so).")
    ap.add_argument("--curve-only", action="store_true", help="Chi ve diem in_curve_zone=1.")
    ap.add_argument("--zoom", type=int, default=19, help="Muc zoom tile ban do (mac dinh 19).")
    ap.add_argument("--route", default=None,
                    help="CSV tuyen (map/gps_log.csv) de quy enc (s,d) -> lat/lon ve len ban "
                         "do; rong = tu tim trong package. Bo qua neu khong tim thay.")
    ap.add_argument("--out", default=None, help="Luu PNG thay vi mo cua so.")
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

    # Mask curve tính 1 lần từ in_zone GỐC — KHÔNG cắt mảng in-place (mọi mảng
    # giữ nguyên độ dài, tránh lệch chiều khi index). --curve-only chỉ thêm
    # điều kiện in_zone vào từng phép vẽ bên dưới.
    curve_mask = in_zone >= 0.5

    # Bản đồ: giữ điểm lat/lon hợp lệ; --curve-only thì thêm trong cua.
    map_sel = np.isfinite(lat) & np.isfinite(lon)
    if args.curve_only:
        map_sel &= curve_mask
        print(f"--curve-only: giu {int(map_sel.sum())}/{len(map_sel)} diem tren ban do")
    if map_sel.sum() < 1:
        sys.exit("Khong co diem lat/lon hop le (hoac khong co diem trong cua voi --curve-only).")
    ex, ey = to_mercator(lat[map_sel], lon[map_sel])
    zone_ok = curve_mask[map_sel]

    try:
        import matplotlib
        if args.out:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        sys.exit("Thieu matplotlib. Cai: pip install matplotlib")

    have_ds = s_m is not None and d_m is not None
    ncols = 2 if have_ds else 1
    fig, axes = plt.subplots(1, ncols, figsize=(8 * ncols, 7))
    ax_map = axes[0] if have_ds else axes

    # --- Bản đồ nền ---
    # GPS thô vẽ TRƯỚC (dưới cùng) = chấm nhỏ, mờ, để đường EKF nổi lên trên.
    raw_sel = np.isfinite(r_lat) & np.isfinite(r_lon)
    if args.curve_only:
        raw_sel &= curve_mask
    if raw_sel.any():
        rx, ry = to_mercator(r_lat[raw_sel], r_lon[raw_sel])
        ax_map.scatter(rx, ry, c="#1f77b4", s=3, alpha=0.35, zorder=3, label="GPS tho (raw)")

    # Encoder thuan (dead-reckoning, route_ekf_enc — khong bao gio correct_gps)
    # — chi co o CSV moi (enc_lat/enc_lon, xem Gps/node.py); CSV cu bo qua.
    enc_sel = np.isfinite(e_lat) & np.isfinite(e_lon)
    if args.curve_only:
        enc_sel &= curve_mask
    if enc_sel.any():
        qx, qy = to_mercator(e_lat[enc_sel], e_lon[enc_sel])
        ax_map.plot(qx, qy, "--", color="#9467bd", lw=1.4, alpha=0.8, zorder=4,
                    label="Encoder thuan (dead-reckoning)")

    # RouteEKF localization: ĐƯỜNG rõ nét. Ngoài cua xám đậm, trong cua cam dày.
    ax_map.plot(ex, ey, "-", color="#333333", lw=1.8, zorder=5, label="RouteEKF (ngoai cua)")
    if zone_ok.any():
        ax_map.plot(ex[zone_ok], ey[zone_ok], "-", color="#ff7f0e", lw=3.0, zorder=6,
                    label="RouteEKF (trong cua)")
    ax_map.scatter(ex[0], ey[0], c="green", s=70, marker="o", zorder=7, label="start")
    ax_map.scatter(ex[-1], ey[-1], c="red", s=70, marker="s", zorder=7, label="end")

    # Ground truth RTK (tuỳ chọn, chỉ vẽ).
    if args.gt:
        gt_rows = load_run(args.gt)
        gfld = gt_rows[0].keys()
        g_lat = _floats(gt_rows, _find_col(gfld, LAT_KEYS))
        g_lon = _floats(gt_rows, _find_col(gfld, LON_KEYS))
        gok = np.isfinite(g_lat) & np.isfinite(g_lon)
        gx, gy = to_mercator(g_lat[gok], g_lon[gok])
        ax_map.plot(gx, gy, "-", color="#2ca02c", lw=2.5, zorder=3, label="Ground truth RTK")

    ax_map.set_aspect("equal", adjustable="datalim")
    ax_map.set_axis_off()
    ax_map.set_title("Quy dao tren ban do nen (OSM)")
    ax_map.legend(loc="best", fontsize=8)

    try:
        import contextily as ctx
        ctx.add_basemap(ax_map, crs=WEB_MERCATOR,
                        source=ctx.providers.OpenStreetMap.Mapnik, zoom=args.zoom)
    except ModuleNotFoundError:
        print("(Thieu contextily -> ve khong co ban do nen. Cai: pip install contextily)")
    except Exception as exc:
        print(f"(Khong tai duoc tile ban do, ve khong nen: {exc})")

    # --- d theo s (route frame): RouteEKF vs encoder thuần ---
    if have_ds:
        ax_ds = axes[1]
        # curve_mask/s_m/d_m đều full length (không cắt in-place) -> index khớp.
        ds_mask = curve_mask if args.curve_only else np.ones(len(s_m), bool)
        ax_ds.plot(s_m[ds_mask], d_m[ds_mask], color="#ff7f0e", lw=1.6, label="RouteEKF d (fused)")
        if enc_s is not None and enc_d is not None:
            ax_ds.plot(enc_s[ds_mask], enc_d[ds_mask], color="#1f77b4", lw=1.4, ls="--",
                       label="encoder thuan d (khong GPS/vision)")
        ax_ds.axhline(0.0, color="0.7", lw=0.8)
        ax_ds.set_xlabel("s [m] (doc tuyen)")
        ax_ds.set_ylabel("d [m] (lech ngang, +=phai)")
        ax_ds.set_title("Lech ngang theo tuyen — RouteEKF vs encoder thuan")
        ax_ds.grid(True, alpha=0.3)
        ax_ds.legend(loc="best", fontsize=9)

    fig.tight_layout()
    print(f"Doc {len(rows)} dong tu {args.run_csv} "
          f"({int(zone_ok.sum())} diem trong cua)")
    if args.out:
        fig.savefig(args.out, dpi=130)
        print(f"Da luu -> {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
