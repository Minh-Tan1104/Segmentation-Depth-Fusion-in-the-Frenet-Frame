from __future__ import annotations

import argparse
import os
import tempfile
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

# Matplotlib needs a writable cache dir in this environment.
_MPLCONFIGDIR = Path(tempfile.gettempdir()) / "matplotlib"
_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR))

try:
    import contextily as ctx
    import matplotlib.pyplot as plt
    from pyproj import Transformer
    from pyubx2 import UBXReader
    from serial import Serial, SerialException
except ImportError as exc:
    missing = exc.name or "unknown"
    raise SystemExit(
        "Missing dependency '%s'. Install with: "
        "pip install pyserial pyubx2 matplotlib contextily pyproj" % missing
    ) from exc

from gps_reader import GpsSample, extract_sample
from map_matcher import MatchResult, RouteMapMatcher

WGS84 = "EPSG:4326"
WEB_MERCATOR = "EPSG:3857"

# Tuyến ghi sẵn mặc định (map/gps_path_2m.csv cạnh thư mục Gps/).
DEFAULT_ROUTE_CSV = Path(__file__).resolve().parent.parent / "map" / "gps_log.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read UBX GNSS data and show a live marker on a contextily map."
    )
    parser.add_argument("--port", default="/dev/ttyUSB0", help="Serial device path.")
    parser.add_argument("--baud", type=int, default=230400, help="Serial baud rate.")
    parser.add_argument(
        "--timeout",
        type=float,
        default=1.0,
        help="Serial read timeout in seconds.",
    )
    parser.add_argument(
        "--radius-m",
        type=float,
        default=120.0,
        help="Half-width of the displayed map around the current point.",
    )
    parser.add_argument(
        "--trail-length",
        type=int,
        default=500,
        help="Maximum number of recent GPS points to keep in the trail.",
    )
    parser.add_argument(
        "--zoom",
        type=int,
        default=19,
        help="Basemap zoom passed to contextily.",
    )
    parser.add_argument(
        "--route",
        default=str(DEFAULT_ROUTE_CSV),
        help="CSV lat,lon cua tuyen ghi san de map-matching (rong = tat).",
    )
    return parser.parse_args()


class LiveCtxMap:
    def __init__(
        self,
        radius_m: float,
        trail_length: int,
        zoom: int,
        route_lonlat: tuple[list[float], list[float]] | None = None,
    ) -> None:
        self.radius_m = max(20.0, float(radius_m))
        self.zoom = int(zoom)
        self.transformer = Transformer.from_crs(
            WGS84, WEB_MERCATOR, always_xy=True
        )
        self.xs: deque[float] = deque(maxlen=max(2, int(trail_length)))
        self.ys: deque[float] = deque(maxlen=max(2, int(trail_length)))
        self.center_x: float | None = None
        self.center_y: float | None = None
        self.closed = False

        # Tuyến ghi sẵn (nếu có) — chiếu 1 lần sang Web Mercator để vẽ nền.
        self.route_xy: tuple[list[float], list[float]] | None = None
        if route_lonlat is not None:
            rx, ry = self.transformer.transform(route_lonlat[0], route_lonlat[1])
            self.route_xy = (list(rx), list(ry))

        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        if getattr(self.fig.canvas, "manager", None) is not None:
            self.fig.canvas.manager.set_window_title("UBX GNSS Map")
        self.fig.canvas.mpl_connect("close_event", self._on_close)

        self.line = None
        self.point = None
        self.match_point = None
        self.status = None
        self.ax.set_title("UBX GNSS live map")
        self.ax.set_axis_off()
        self.ax.text(
            0.5,
            0.5,
            "Waiting for first GNSS fix...",
            transform=self.ax.transAxes,
            ha="center",
            va="center",
            fontsize=11,
            bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "#777777"},
        )

    def _on_close(self, _event: object) -> None:
        self.closed = True

    def _reset_axes(self, center_x: float, center_y: float) -> None:
        self.center_x = center_x
        self.center_y = center_y

        self.ax.clear()
        self.ax.set_xlim(center_x - self.radius_m, center_x + self.radius_m)
        self.ax.set_ylim(center_y - self.radius_m, center_y + self.radius_m)
        self.ax.set_aspect("equal", adjustable="box")
        self.ax.set_title("UBX GNSS live map")
        self.ax.set_axis_off()

        try:
            ctx.add_basemap(
                self.ax,
                crs=WEB_MERCATOR,
                source=ctx.providers.OpenStreetMap.Mapnik,
                zoom=self.zoom,
            )
        except Exception as exc:
            self.ax.text(
                0.5,
                0.5,
                f"Basemap load failed:\n{exc}",
                transform=self.ax.transAxes,
                ha="center",
                va="center",
                fontsize=10,
                bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "black"},
            )

        if self.route_xy is not None:
            self.ax.plot(
                self.route_xy[0],
                self.route_xy[1],
                color="#7b1fa2",
                linewidth=2.5,
                linestyle="--",
                alpha=0.8,
                zorder=3,
            )
        (self.line,) = self.ax.plot(
            [],
            [],
            color="#00acc1",
            linewidth=2.0,
            zorder=4,
        )
        self.point = self.ax.scatter(
            [],
            [],
            s=90,
            c="#d32f2f",
            edgecolors="white",
            linewidths=1.5,
            zorder=5,
        )
        # Điểm chiếu của xe lên tuyến ghi sẵn (kết quả map-matching).
        self.match_point = self.ax.scatter(
            [],
            [],
            s=70,
            c="#7b1fa2",
            marker="s",
            edgecolors="white",
            linewidths=1.0,
            zorder=5,
        )
        self.status = self.ax.text(
            0.02,
            0.98,
            "",
            transform=self.ax.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "#777777"},
        )

    def update(self, sample: GpsSample, match: MatchResult | None = None) -> None:
        x, y = self.transformer.transform(sample.lon, sample.lat)
        self.xs.append(x)
        self.ys.append(y)

        if self._needs_recenter(x, y):
            self._reset_axes(x, y)

        self.line.set_data(list(self.xs), list(self.ys))
        self.point.set_offsets([[x, y]])
        if match is not None:
            mx, my = self.transformer.transform(match.lon, match.lat)
            self.match_point.set_offsets([[mx, my]])
        sats = "-" if sample.satellites is None else str(sample.satellites)
        acc = "-" if sample.h_acc_m is None else f"{sample.h_acc_m:.1f}m"
        lines = [
            f"{sample.identity}  fix={sample.fix_label}  sats={sats}  hAcc={acc}",
            f"lat={sample.lat:.7f}",
            f"lon={sample.lon:.7f}",
        ]
        if match is not None:
            lines.append(
                f"s={match.s_m:.1f}m ({match.progress * 100.0:.0f}%)"
                f"  xtrack={match.cross_track_m:.1f}m"
            )
        self.status.set_text("\n".join(lines))

        self.fig.canvas.draw_idle()
        plt.pause(0.001)

    def _needs_recenter(self, x: float, y: float) -> bool:
        if self.center_x is None or self.center_y is None:
            return True

        limit = self.radius_m * 0.35
        return (
            abs(x - self.center_x) > limit or abs(y - self.center_y) > limit
        )


def main() -> None:
    args = parse_args()
    plt.ion()

    try:
        stream = Serial(args.port, args.baud, timeout=args.timeout)
    except SerialException as exc:
        raise SystemExit(f"Cannot open serial port {args.port}: {exc}") from exc

    # Map-matching lên tuyến ghi sẵn (nếu có file): GPS chỉ để biết xe đang ở
    # đâu dọc tuyến (s_match), không tham gia điều khiển lệch ngang.
    matcher: RouteMapMatcher | None = None
    if args.route:
        route_path = Path(args.route)
        if route_path.is_file():
            matcher = RouteMapMatcher(route_path)
            print(
                f"Route loaded: {route_path.name}, "
                f"{matcher.total_length_m:.0f} m, {len(matcher.route_lat)} points",
                flush=True,
            )
        else:
            print(f"Route CSV khong ton tai, tat map-matching: {route_path}", flush=True)

    viewer = LiveCtxMap(
        radius_m=args.radius_m,
        trail_length=args.trail_length,
        zoom=args.zoom,
        route_lonlat=(matcher.route_lon, matcher.route_lat) if matcher else None,
    )
    reader = UBXReader(stream)

    last_status_print = 0.0
    status_interval = 1.0

    try:
        while not viewer.closed:
            _, msg = reader.read()
            if msg is None:
                now = time.monotonic()
                if now - last_status_print >= status_interval:
                    print("No data from GPS (mat ket noi hoac chua nhan tin hieu)...", flush=True)
                    last_status_print = now
                plt.pause(0.001)
                continue

            sample = extract_sample(msg)
            if sample is None:
                now = time.monotonic()
                if now - last_status_print >= status_interval:
                    print("GPS connected, chua co fix (dang cho ve tinh)...", flush=True)
                    last_status_print = now
                continue

            match = None
            match_note = ""
            if matcher is not None:
                match, reason = matcher.update(
                    sample.lat,
                    sample.lon,
                    fix_type=sample.fix_type,
                    h_acc_m=sample.h_acc_m,
                )
                if match is not None:
                    match_note = (
                        f", s={match.s_m:.1f}m/{matcher.total_length_m:.0f}m"
                        f" ({match.progress * 100.0:.0f}%)"
                        f", xtrack={match.cross_track_m:.1f}m"
                    )
                else:
                    match_note = f", match-reject: {reason}"
                    match = matcher.last_match  # vẫn vẽ s cũ còn tin được

            print(
                f"{sample.identity}: lat={sample.lat:.7f}, "
                f"lon={sample.lon:.7f}, fix={sample.fix_label}, "
                f"sats={sample.satellites}{match_note}",
                flush=True,
            )
            viewer.update(sample, match)
    except KeyboardInterrupt:
        pass
    finally:
        stream.close()
        plt.ioff()
        plt.close("all")


if __name__ == "__main__":
    main()
