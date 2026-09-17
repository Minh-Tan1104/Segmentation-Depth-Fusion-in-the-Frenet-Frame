#!/usr/bin/env python3
"""Vẽ waypoint (lat, lon) của 1 file CSV tuyến lên bản đồ nền thật (contextily,
tile CartoDB) — CHỈ để soi trực quan file route, không so sánh với run nào cả.

CSV cần tối thiểu 2 cột lat/lon (tự dò tên cột, xem LAT_KEYS/LON_KEYS bên
dưới) — dùng được cho map/gps_path_2m.csv, map/gps_log.csv, hoặc bất kỳ file
run nào của gps_node (log_latlon_enable).

Dùng tile CartoDB (KHÔNG dùng OpenStreetMap.Mapnik mặc định của contextily)
— tile.openstreetmap.org trả 403 "Access blocked" khi request từ môi trường
này (chặn theo policy/User-Agent), CartoDB tải bình thường. Mặc định
"voyager" (đường phố có màu, giống Google Maps); đổi bằng --basemap.

Cần mạng để tải tile lúc chạy. Thuần Python, KHÔNG rclpy.
    pip install matplotlib contextily pyproj

Curve zone (tô cam + khoanh nét đứt) dò TRỰC TIẾP từ hình học tuyến qua
RouteMapMatcher.detect_curve_zones() — đúng cơ chế gps_node dùng để quyết
định lúc nào chặn GPS/anchor vision (xem Gps/map_matcher.py). Ngưỡng mặc
định khớp config/rl_car_params.yaml (curve_zone_curvature_thresh=0.1,
dilate=[10,5]/[3,5]) — tắt bằng --no-curve-zone.

Dùng:
    python3 scripts/plot_waypoints_map.py map/gps_path_2m.csv
    python3 scripts/plot_waypoints_map.py map/gps_log.csv --out waypoints.png
    python3 scripts/plot_waypoints_map.py map/gps_path_2m.csv --no-line --zoom 20
    python3 scripts/plot_waypoints_map.py map/gps_path_2m.csv --no-curve-zone
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

try:
    from pyproj import Transformer
except ModuleNotFoundError:
    sys.exit("Thieu pyproj. Cai: pip install pyproj")

try:
    import contextily as ctx
except ModuleNotFoundError:
    sys.exit("Thieu contextily. Cai: pip install contextily")

# Cho phep chay truc tiep `python3 scripts/plot_waypoints_map.py` khong can
# cai package qua colcon — giong plot_route_compare.py.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from Gps.map_matcher import RouteMapMatcher  # noqa: E402

WGS84 = "EPSG:4326"
WEB_MERCATOR = "EPSG:3857"
LAT_KEYS = ("lat", "latitude", "y")
LON_KEYS = ("lon", "lng", "long", "longitude", "x")

# Khop config/rl_car_params.yaml: curve_zone_curvature_thresh / dilate_before_m
# / dilate_after_m — dung production defaults de zone hien thi dung y he
# gps_node se thay khi chay that.
CURVE_ZONE_CURVATURE_THRESH = 0.1
CURVE_ZONE_DILATE_BEFORE_M = [10.0, 5.0]
CURVE_ZONE_DILATE_AFTER_M = [3.0, 5.0]

BASEMAPS = {
    "voyager": lambda ctx: ctx.providers.CartoDB.Voyager,
    "positron": lambda ctx: ctx.providers.CartoDB.Positron,
    "satellite": lambda ctx: ctx.providers.Esri.WorldImagery,
    "topo": lambda ctx: ctx.providers.OpenTopoMap,
}


def contiguous_true(mask: np.ndarray) -> list[tuple[int, int]]:
    """Danh sach (i0, i1) cac doan True lien tiep trong mask (i1 exclusive)."""
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(idx) > 1)
    starts = np.concatenate([[0], breaks + 1])
    ends = np.concatenate([breaks + 1, [idx.size]])
    return [(int(idx[s]), int(idx[e - 1]) + 1) for s, e in zip(starts, ends)]


def _find_col(fieldnames, keys):
    low = {f.lower().strip(): f for f in fieldnames}
    for k in keys:
        if k in low:
            return low[k]
    return None


def load_waypoints(path: str) -> tuple[np.ndarray, np.ndarray]:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        sys.exit(f"CSV rong: {path}")
    fld = rows[0].keys()
    lat_col = _find_col(fld, LAT_KEYS)
    lon_col = _find_col(fld, LON_KEYS)
    if lat_col is None or lon_col is None:
        sys.exit(f"Khong tim thay cot lat/lon trong {path} (co: {list(fld)})")

    lat, lon = [], []
    for r in rows:
        try:
            la, lo = float(r[lat_col]), float(r[lon_col])
        except (ValueError, TypeError):
            continue
        lat.append(la)
        lon.append(lo)
    if not lat:
        sys.exit(f"Khong doc duoc waypoint hop le nao tu {path}")
    return np.asarray(lat), np.asarray(lon)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Ve waypoint (lat,lon) cua 1 file CSV tuyen len ban do nen (contextily)."
    )
    ap.add_argument("csv", help="File CSV tuyen (can cot lat/lon).")
    ap.add_argument("--no-line", action="store_true",
                     help="Chi ve cham waypoint, khong noi duong.")
    ap.add_argument("--zoom", type=int, default=19, help="Muc zoom tile (mac dinh 19).")
    ap.add_argument("--basemap", choices=sorted(BASEMAPS), default="voyager",
                     help="Kieu nen ban do (mac dinh voyager).")
    ap.add_argument("--no-curve-zone", action="store_true",
                     help="Khong do/highlight curve zone.")
    ap.add_argument("--curve-thresh", type=float, default=CURVE_ZONE_CURVATURE_THRESH,
                     help=f"Nguong curvature [1/m] coi la cua (mac dinh {CURVE_ZONE_CURVATURE_THRESH}).")
    ap.add_argument("--trim-to-curve", action="store_true",
                     help="Cat bo doan thang truoc curve zone dau tien — diem"
                          " 'start' doi sang dau zone 0 (can bat curve zone).")
    ap.add_argument("--out", default=None, help="Luu PNG thay vi mo cua so.")
    args = ap.parse_args()

    import matplotlib
    if args.out:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    lat, lon = load_waypoints(args.csv)
    tf = Transformer.from_crs(WGS84, WEB_MERCATOR, always_xy=True)
    x, y = tf.transform(lon, lat)

    zones: list[tuple[float, float]] = []
    in_zone = np.zeros(len(lat), dtype=bool)
    if not args.no_curve_zone:
        matcher = RouteMapMatcher(args.csv)
        zones = matcher.detect_curve_zones(
            curvature_thresh=args.curve_thresh,
            dilate_before_m=CURVE_ZONE_DILATE_BEFORE_M,
            dilate_after_m=CURVE_ZONE_DILATE_AFTER_M,
        )
        s_vals = np.array([
            matcher.project_xy(*matcher.to_local(la, lo))[0] for la, lo in zip(lat, lon)
        ])
        for a, b in zones:
            in_zone |= (s_vals >= a) & (s_vals <= b)

        if args.trim_to_curve:
            if not zones:
                sys.exit("--trim-to-curve nhung khong do duoc curve zone nao.")
            start_i = int(np.argmax(s_vals >= zones[0][0]))
            lat, lon = lat[start_i:], lon[start_i:]
            x, y = x[start_i:], y[start_i:]
            in_zone = in_zone[start_i:]
            print(f"Cat {start_i} waypoint dau (truoc s={zones[0][0]:.1f}m) — "
                  f"con {len(lat)} diem.")

    fig, ax = plt.subplots(figsize=(9, 9))
    if not args.no_line:
        ax.plot(x, y, "-", color="#1f77b4", lw=1.5, alpha=0.8, zorder=2)
    ax.scatter(x[~in_zone], y[~in_zone], c="#d62728", s=14, zorder=3,
               edgecolors="white", linewidths=0.4, label="waypoint")
    if in_zone.any():
        ax.scatter(x[in_zone], y[in_zone], c="#ff7f0e", s=18, zorder=4,
                   edgecolors="white", linewidths=0.4, label="waypoint (curve zone)")
        circ_label = True
        for i0, i1 in contiguous_true(in_zone):
            cx, cy = np.mean(x[i0:i1]), np.mean(y[i0:i1])
            rad = np.max(np.hypot(x[i0:i1] - cx, y[i0:i1] - cy)) + 3.0
            ax.add_patch(Circle((cx, cy), rad, fill=False, ls="--", lw=1.8,
                                 edgecolor="#17becf", zorder=5,
                                 label="Curve zone" if circ_label else None))
            circ_label = False
    ax.scatter(x[0], y[0], c="green", s=110, marker="o", edgecolors="k", zorder=6, label="start")
    ax.scatter(x[-1], y[-1], c="red", s=110, marker="s", edgecolors="k", zorder=6, label="end")

    ax.set_aspect("equal", adjustable="datalim")
    ax.set_axis_off()
    ax.legend(loc="best", fontsize=9)
    ax.set_title(f"Waypoints — {args.csv} ({len(x)} points, {len(zones)} curve zone(s))")
    ctx.add_basemap(ax, crs=WEB_MERCATOR, source=BASEMAPS[args.basemap](ctx), zoom=args.zoom)
    fig.tight_layout()

    print(f"Doc {len(x)} waypoint tu {args.csv}")
    if zones:
        print(f"Curve zone (curvature_thresh={args.curve_thresh}):")
        for i, (a, b) in enumerate(zones):
            print(f"  zone {i}: s=[{a:.1f}, {b:.1f}]m  (dai {b - a:.1f}m)")

    if args.out:
        fig.savefig(args.out, dpi=150, bbox_inches="tight")
        print(f"Da luu -> {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
