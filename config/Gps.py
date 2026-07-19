from __future__ import annotations

import argparse
import math
import os
import tempfile
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
    from pyubx2 import SET, UBXMessage, UBXReader
    from serial import Serial, SerialException
except ImportError as exc:
    missing = exc.name or "unknown"
    raise SystemExit(
        "Missing dependency '%s'. Install with: "
        "pip install pyserial pyubx2 matplotlib contextily pyproj" % missing
    ) from exc

WGS84 = "EPSG:4326"
WEB_MERCATOR = "EPSG:3857"

FIX_TYPE_LABELS = {
    0: "no-fix",
    1: "dead-reckoning",
    2: "2D",
    3: "3D",
    4: "GNSS+DR",
    5: "time-only",
}


@dataclass(slots=True)
class GpsSample:
    identity: str
    lat: float
    lon: float
    fix_type: int | None
    satellites: int | None

    @property
    def fix_label(self) -> str:
        if self.fix_type is None:
            return "unknown"
        return FIX_TYPE_LABELS.get(self.fix_type, str(self.fix_type))


@dataclass(slots=True)
class RawMeasurement:
    gnss_id: int
    sv_id: int
    sig_id: int
    pr_mes: float
    cp_mes: float
    do_mes: float
    cno: int
    locktime: int
    pr_valid: bool
    cp_valid: bool


@dataclass(slots=True)
class RawEpoch:
    rcv_tow: float
    week: int
    leap_s: int
    num_meas: int
    measurements: list[RawMeasurement]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read UBX GNSS data and show a live marker on a contextily map."
    )
    parser.add_argument("--port", default="/dev/ttyUSB1", help="Serial device path.")
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
        "--enable-rawx",
        dest="enable_rawx",
        action="store_true",
        default=True,
        help="Send a UBX CFG-MSG on startup to enable RXM-RAWX raw "
        "measurement output (default: enabled).",
    )
    parser.add_argument(
        "--no-enable-rawx",
        dest="enable_rawx",
        action="store_false",
        help="Do not send the CFG-MSG that enables RXM-RAWX output.",
    )
    return parser.parse_args()


def build_enable_rawx_message() -> UBXMessage:
    """UBX CFG-MSG enabling RXM-RAWX (class 0x02, id 0x15) on UART1/USB."""
    return UBXMessage(
        "CFG",
        "CFG-MSG",
        SET,
        msgClass=0x02,
        msgID=0x15,
        rateUART1=1,
        rateUSB=1,
    )


def extract_sample(msg: object) -> GpsSample | None:
    lat = getattr(msg, "lat", None)
    lon = getattr(msg, "lon", None)
    if lat is None or lon is None:
        return None

    lat = float(lat)
    lon = float(lon)
    if not math.isfinite(lat) or not math.isfinite(lon):
        return None
    if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
        return None

    gnss_fix_ok = getattr(msg, "gnssFixOk", None)
    fix_type = getattr(msg, "fixType", None)
    if gnss_fix_ok is not None and not bool(gnss_fix_ok):
        return None
    if fix_type is not None and int(fix_type) <= 0:
        return None

    satellites = getattr(msg, "numSV", getattr(msg, "numSvs", None))
    identity = str(getattr(msg, "identity", type(msg).__name__))
    return GpsSample(
        identity=identity,
        lat=lat,
        lon=lon,
        fix_type=None if fix_type is None else int(fix_type),
        satellites=None if satellites is None else int(satellites),
    )


def extract_raw_epoch(msg: object) -> RawEpoch | None:
    identity = str(getattr(msg, "identity", type(msg).__name__))
    if identity != "RXM-RAWX":
        return None

    num_meas = int(getattr(msg, "numMeas", 0))
    measurements: list[RawMeasurement] = []
    for i in range(1, num_meas + 1):
        suffix = f"_{i:02d}"
        measurements.append(
            RawMeasurement(
                gnss_id=int(getattr(msg, f"gnssId{suffix}", -1)),
                sv_id=int(getattr(msg, f"svId{suffix}", -1)),
                sig_id=int(getattr(msg, f"sigId{suffix}", -1)),
                pr_mes=float(getattr(msg, f"prMes{suffix}", math.nan)),
                cp_mes=float(getattr(msg, f"cpMes{suffix}", math.nan)),
                do_mes=float(getattr(msg, f"doMes{suffix}", math.nan)),
                cno=int(getattr(msg, f"cno{suffix}", 0)),
                locktime=int(getattr(msg, f"locktime{suffix}", 0)),
                pr_valid=bool(getattr(msg, f"prValid{suffix}", 0)),
                cp_valid=bool(getattr(msg, f"cpValid{suffix}", 0)),
            )
        )

    return RawEpoch(
        rcv_tow=float(getattr(msg, "rcvTow", math.nan)),
        week=int(getattr(msg, "week", -1)),
        leap_s=int(getattr(msg, "leapS", -1)),
        num_meas=num_meas,
        measurements=measurements,
    )


class LiveCtxMap:
    def __init__(self, radius_m: float, trail_length: int, zoom: int) -> None:
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

        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        if getattr(self.fig.canvas, "manager", None) is not None:
            self.fig.canvas.manager.set_window_title("UBX GNSS Map")
        self.fig.canvas.mpl_connect("close_event", self._on_close)

        self.line = None
        self.point = None
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

    def update(self, sample: GpsSample) -> None:
        x, y = self.transformer.transform(sample.lon, sample.lat)
        self.xs.append(x)
        self.ys.append(y)

        if self._needs_recenter(x, y):
            self._reset_axes(x, y)

        self.line.set_data(list(self.xs), list(self.ys))
        self.point.set_offsets([[x, y]])
        sats = "-" if sample.satellites is None else str(sample.satellites)
        self.status.set_text(
            "\n".join(
                [
                    f"{sample.identity}  fix={sample.fix_label}  sats={sats}",
                    f"lat={sample.lat:.7f}",
                    f"lon={sample.lon:.7f}",
                ]
            )
        )

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

    if args.enable_rawx:
        stream.write(build_enable_rawx_message().serialize())
        stream.flush()

    viewer = LiveCtxMap(
        radius_m=args.radius_m,
        trail_length=args.trail_length,
        zoom=args.zoom,
    )
    reader = UBXReader(stream)

    try:
        while not viewer.closed:
            raw_data, msg = reader.read()
            if msg is None:
                if raw_data:
                    print(f"[raw bytes] {raw_data.hex()}", flush=True)
                plt.pause(0.001)
                continue

            raw_epoch = extract_raw_epoch(msg)
            if raw_epoch is not None:
                print(
                    f"RXM-RAWX: week={raw_epoch.week} "
                    f"rcvTow={raw_epoch.rcv_tow:.3f} "
                    f"leapS={raw_epoch.leap_s} numMeas={raw_epoch.num_meas}",
                    flush=True,
                )
                for m in raw_epoch.measurements:
                    print(
                        f"  gnssId={m.gnss_id} svId={m.sv_id} sigId={m.sig_id} "
                        f"prMes={m.pr_mes:.3f} cpMes={m.cp_mes:.3f} "
                        f"doMes={m.do_mes:.3f} cno={m.cno} "
                        f"locktime={m.locktime} prValid={m.pr_valid} "
                        f"cpValid={m.cp_valid}",
                        flush=True,
                    )
                continue

            sample = extract_sample(msg)
            if sample is None:
                identity = str(getattr(msg, "identity", type(msg).__name__))
                print(f"[{identity}] {msg}", flush=True)
                continue

            print(
                f"{sample.identity}: lat={sample.lat:.7f}, "
                f"lon={sample.lon:.7f}, fix={sample.fix_label}, "
                f"sats={sample.satellites}",
                flush=True,
            )
            viewer.update(sample)
    except KeyboardInterrupt:
        pass
    finally:
        stream.close()
        plt.ioff()
        plt.close("all")


if __name__ == "__main__":
    main()
