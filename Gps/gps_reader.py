"""Đọc + validate 1 bản tin UBX NAV-PVT thành GpsSample.

Tách riêng khỏi Gps.py (script xem map độc lập, phụ thuộc matplotlib/
contextily) để Gps/node.py (ROS2, không cần vẽ gì) dùng chung logic parse mà
không phải kéo theo mấy dependency đồ hoạ đó.

Thuần Python, chỉ cần pyserial + pyubx2.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

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
    h_acc_m: float | None  # sai số ngang ước lượng từ receiver (NAV-PVT hAcc) [m]

    @property
    def fix_label(self) -> str:
        if self.fix_type is None:
            return "unknown"
        return FIX_TYPE_LABELS.get(self.fix_type, str(self.fix_type))


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
    h_acc = getattr(msg, "hAcc", None)  # NAV-PVT hAcc: mm (pyubx2 không tự scale)
    return GpsSample(
        identity=identity,
        lat=lat,
        lon=lon,
        fix_type=None if fix_type is None else int(fix_type),
        satellites=None if satellites is None else int(satellites),
        h_acc_m=None if h_acc is None else float(h_acc) * 1e-3,
    )
