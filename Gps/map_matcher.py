"""Map-matching GPS fix lên tuyến đường ghi sẵn trong map/gps_path_2m.csv.

Vai trò: GPS CHỈ trả lời câu hỏi "xe đang ở đâu dọc tuyến (s_match)" để biết
sắp tới đoạn nào/ngã rẽ nào — KHÔNG tham gia điều khiển lệch ngang (d/psi vẫn
chạy ego-frame bằng camera + encoder như cũ).

Ưu tiên chính xác hơn nhanh: fix phải qua 3 lớp gate mới được cập nhật s_match,
fix rớt gate thì giữ nguyên s_match cũ (thà đứng yên trên map còn hơn nhảy sai):
  1. Chất lượng fix:  fixType >= 3 (3D) và hAcc <= max_h_acc_m.
  2. Khoảng cách tới tuyến: điểm chiếu vuông góc không quá max_cross_track_m.
  3. Liên tục: s_match mới không được nhảy quá xa s cũ (giới hạn theo
     max_speed_mps * dt + biên nhiễu), chặn outlier do multipath.

Thuần Python/numpy + pyproj, không import rclpy — GPS đang test độc lập.
"""
from __future__ import annotations

import csv
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    from pyproj import Transformer
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Missing dependency 'pyproj'. Install with: pip install pyproj") from exc

WGS84 = "EPSG:4326"


@dataclass(frozen=True)
class MatchResult:
    """Kết quả 1 lần match thành công."""

    s_m: float             # quãng đường dọc tuyến tính từ điểm đầu CSV [m]
    cross_track_m: float   # khoảng cách vuông góc từ fix tới tuyến [m] (chỉ để sanity-check)
    seg_index: int         # index đoạn polyline được match (điểm i -> i+1)
    lat: float             # toạ độ điểm chiếu trên tuyến
    lon: float
    progress: float        # s_m / tổng chiều dài tuyến, 0..1


@dataclass
class MatcherConfig:
    min_fix_type: int = 3          # chỉ nhận 3D fix trở lên
    max_h_acc_m: float = 5.0       # hAcc ước lượng của receiver phải dưới ngưỡng này
    max_cross_track_m: float = 8.0  # fix chiếu xa tuyến hơn mức này -> coi là outlier
    max_speed_mps: float = 5.0     # tốc độ tối đa giả định của xe, để giới hạn bước nhảy s
    jump_margin_m: float = 6.0     # biên nhiễu cộng thêm khi kiểm tra bước nhảy s
    search_back_m: float = 10.0    # cửa sổ tìm kiếm lùi quanh s cũ (xe không đi lùi xa)
    stale_reset_s: float = 30.0    # mất fix lâu hơn mức này -> bỏ cửa sổ, tìm toàn tuyến lại
    # Tuyến CSV ghi bằng GPS nên toạ độ răng cưa ~m -> heading giữa các điểm
    # 2m lệch vài độ. Làm mượt (moving average) trước khi dựng polyline để
    # heading_at() phản ánh hướng ĐƯỜNG THẬT chứ không phải nhiễu lúc ghi —
    # RouteEKF neo heading vào đây nên nhiễu này đi thẳng vào trôi d khi cua.
    smooth_window: int = 5         # số điểm moving average (1 = tắt)


class RouteMapMatcher:
    """Nạp tuyến từ CSV (cột lat,lon) và match từng fix GPS lên tuyến đó."""

    def __init__(self, csv_path: str | Path, config: MatcherConfig | None = None) -> None:
        self.config = config or MatcherConfig()
        lats, lons = self._read_csv(Path(csv_path))
        if len(lats) < 2:
            raise ValueError(f"Route CSV can it nhat 2 diem, doc duoc {len(lats)}")
        self.route_lat: list[float] = lats.tolist()  # để viewer vẽ tuyến lên map
        self.route_lon: list[float] = lons.tolist()

        # Hệ mét cục bộ tự định nghĩa: azimuthal equidistant đặt gốc tại tâm
        # tuyến — CSV lẫn fix sống đều chiếu qua đây nên luôn nhất quán,
        # không phụ thuộc cột east_m/north_m cũ trong file.
        lat0 = float(np.mean(lats))
        lon0 = float(np.mean(lons))
        self._to_local = Transformer.from_crs(
            WGS84, f"+proj=aeqd +lat_0={lat0} +lon_0={lon0} +units=m", always_xy=True
        )
        self._to_wgs84 = Transformer.from_crs(
            f"+proj=aeqd +lat_0={lat0} +lon_0={lon0} +units=m", WGS84, always_xy=True
        )

        xs, ys = self._to_local.transform(lons, lats)
        xs = self._smooth(np.asarray(xs), self.config.smooth_window)
        ys = self._smooth(np.asarray(ys), self.config.smooth_window)
        self._pts = np.column_stack([xs, ys])                    # (N, 2)
        self._seg_vec = self._pts[1:] - self._pts[:-1]           # (N-1, 2)
        self._seg_len = np.linalg.norm(self._seg_vec, axis=1)    # (N-1,)
        if np.any(self._seg_len <= 0.0):
            raise ValueError("Route CSV co 2 diem lien tiep trung nhau")
        self._s_cum = np.concatenate([[0.0], np.cumsum(self._seg_len)])  # (N,)
        self.total_length_m = float(self._s_cum[-1])

        self._last: MatchResult | None = None
        self._last_time: float | None = None
        self._curv_profile: tuple[np.ndarray, np.ndarray] | None = None  # lazy, xem curvature_at()
        self._curv_profile_key: tuple[int, int] | None = None

    # ------------------------------------------------------------------ #

    @staticmethod
    def _smooth(v: np.ndarray, window: int) -> np.ndarray:
        """Moving average giữ nguyên số điểm (pad mép bằng giá trị biên)."""
        if window <= 1:
            return v
        pad = window // 2
        padded = np.pad(v, pad, mode="edge")
        return np.convolve(padded, np.ones(window) / window, mode="valid")[: len(v)]

    @staticmethod
    def _read_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
        lats: list[float] = []
        lons: list[float] = []
        with path.open(newline="") as f:
            for row in csv.DictReader(f):
                lats.append(float(row["lat"]))
                lons.append(float(row["lon"]))
        return np.asarray(lats), np.asarray(lons)

    @property
    def last_match(self) -> MatchResult | None:
        """s_match gần nhất còn tin được (fix rớt gate không làm mất giá trị này)."""
        return self._last

    @property
    def route_xy_local(self) -> np.ndarray:
        """Polyline tuyến (N,2) trong frame mét cục bộ — dùng để vẽ map panel
        (visualization/logic.py: draw_gps_panel), đã làm mượt như lúc match."""
        return self._pts.copy()

    def reset(self) -> None:
        self._last = None
        self._last_time = None

    # ---- Helper hình học công khai (RouteEKF dùng chung frame local này) ---- #

    def to_local(self, lat: float, lon: float) -> tuple[float, float]:
        """WGS84 -> (x, y) trong frame mét cục bộ của tuyến."""
        x, y = self._to_local.transform(lon, lat)
        return float(x), float(y)

    def to_wgs84(self, x: float, y: float) -> tuple[float, float]:
        """(x, y) local -> (lat, lon)."""
        lon, lat = self._to_wgs84.transform(x, y)
        return float(lat), float(lon)

    def point_at(self, s_m: float) -> tuple[float, float]:
        """Toạ độ local của điểm nằm tại s_m dọc tuyến."""
        return self._point_at(s_m)

    def heading_at(self, s_m: float) -> float:
        """Góc tiếp tuyến của tuyến tại s_m [rad, CCW+, frame local]."""
        s = float(np.clip(s_m, 0.0, self.total_length_m))
        seg_i = int(np.searchsorted(self._s_cum, s, side="right") - 1)
        seg_i = min(seg_i, len(self._seg_len) - 1)
        vx, vy = self._seg_vec[seg_i]
        return math.atan2(vy, vx)

    def project_xy(
        self, x: float, y: float, s_lo: float | None = None, s_hi: float | None = None
    ) -> tuple[float, float, int]:
        """Chiếu điểm local (x, y) lên tuyến, trả (s_m, d_signed_m, seg_index).

        d_signed theo quy ước panel của control/ekf.py: +d = xe lệch sang PHẢI
        so với chiều đi của tuyến.
        """
        lo = 0.0 if s_lo is None else s_lo
        hi = self.total_length_m if s_hi is None else s_hi
        return self._project(x, y, lo, hi)

    def detect_curve_zones(
        self,
        curvature_thresh: float = 0.05,
        dilate_m: float = 4.0,
        n_samples: int = 400,
        smooth_window: int = 7,
    ) -> list[tuple[float, float]]:
        """Tìm các đoạn [s_start, s_end] có độ cong lớn (dùng để trigger chế độ
        "mất vision lúc rẽ": GpsNode chỉ anchor_lateral() từ vision khi NGOÀI
        các đoạn này). curvature_thresh tính bằng rad/m (0.05 ~ bán kính < 20m).
        Kiểm chứng bằng Gps/sim_route_ekf.py trên map/gps_path_2m.csv.
        """
        ss = np.linspace(0.0, self.total_length_m, n_samples)
        headings = np.unwrap([self.heading_at(s) for s in ss])
        ds = ss[1] - ss[0]
        curv = np.abs(np.gradient(headings, ds))
        if smooth_window > 1:
            kernel = np.ones(smooth_window) / smooth_window
            curv = np.convolve(curv, kernel, mode="same")

        zones: list[tuple[float, float]] = []
        in_zone = False
        start = 0.0
        for s, c in zip(ss, curv):
            if c > curvature_thresh and not in_zone:
                in_zone, start = True, s
            elif c <= curvature_thresh and in_zone:
                in_zone = False
                zones.append((max(0.0, start - dilate_m), s + dilate_m))
        if in_zone:
            zones.append((max(0.0, start - dilate_m), self.total_length_m))

        merged: list[tuple[float, float]] = []
        for z in zones:
            if merged and z[0] <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], z[1]))
            else:
                merged.append(z)
        return merged

    def curvature_at(
        self, s_m: float, n_samples: int = 400, smooth_window: int = 7
    ) -> float:
        """Độ cong CÓ DẤU của tuyến tại s_m [rad/m, CCW+ = cua trái, frame local].

        Dùng cho feed-forward lái lúc mất vision (xem Gps/node.py:_tick) —
        khác detect_curve_zones() (lấy abs() để phát hiện zone), hàm này GIỮ
        DẤU để biết lái trái hay phải. Cùng cách lấy mẫu/làm mượt để nhất
        quán với zone (n_samples/smooth_window mặc định khớp nhau); cache lazy
        theo (n_samples, smooth_window) vì tuyến CSV không đổi sau khi load.
        """
        key = (n_samples, smooth_window)
        if self._curv_profile is None or self._curv_profile_key != key:
            ss = np.linspace(0.0, self.total_length_m, n_samples)
            headings = np.unwrap([self.heading_at(s) for s in ss])
            ds = ss[1] - ss[0]
            curv = np.gradient(headings, ds)
            if smooth_window > 1:
                kernel = np.ones(smooth_window) / smooth_window
                curv = np.convolve(curv, kernel, mode="same")
            self._curv_profile = (ss, curv)
            self._curv_profile_key = key

        ss, curv = self._curv_profile
        s = float(np.clip(s_m, 0.0, self.total_length_m))
        return float(np.interp(s, ss, curv))

    # ------------------------------------------------------------------ #

    def update(
        self,
        lat: float,
        lon: float,
        fix_type: int | None = None,
        h_acc_m: float | None = None,
        timestamp: float | None = None,
    ) -> tuple[MatchResult | None, str]:
        """Match 1 fix GPS lên tuyến.

        Trả (result, reason): result=None nghĩa là fix bị loại, reason ghi lý do
        (để in debug); khi đó s_match cũ trong self.last_match vẫn giữ nguyên.
        """
        now = time.monotonic() if timestamp is None else timestamp

        # Gate 1: chất lượng fix từ receiver.
        if fix_type is not None and fix_type < self.config.min_fix_type:
            return None, f"fix_type={fix_type} < {self.config.min_fix_type}"
        if h_acc_m is not None and h_acc_m > self.config.max_h_acc_m:
            return None, f"hAcc={h_acc_m:.1f}m > {self.config.max_h_acc_m}m"

        x, y = self._to_local.transform(lon, lat)

        # Cửa sổ tìm kiếm quanh s cũ (nếu có và chưa quá cũ) — tránh match nhầm
        # sang đoạn tuyến khác chạy gần song song. Mất fix lâu -> tìm toàn tuyến.
        stale = (
            self._last is None
            or self._last_time is None
            or (now - self._last_time) > self.config.stale_reset_s
        )
        if stale:
            lo, hi = 0.0, self.total_length_m
        else:
            dt = max(0.0, now - self._last_time)
            reach = self.config.max_speed_mps * dt + self.config.jump_margin_m
            lo = self._last.s_m - self.config.search_back_m
            hi = self._last.s_m + reach

        s_m, d_signed, seg_idx = self._project(x, y, lo, hi)
        cross = abs(d_signed)

        # Gate 2: fix phải nằm gần tuyến.
        if cross > self.config.max_cross_track_m:
            return None, f"cross_track={cross:.1f}m > {self.config.max_cross_track_m}m"

        # Gate 3: bước nhảy s phải khả thi với tốc độ xe (chỉ khi có match cũ).
        if not stale:
            dt = max(0.0, now - self._last_time)
            max_jump = self.config.max_speed_mps * dt + self.config.jump_margin_m
            if abs(s_m - self._last.s_m) > max_jump:
                return None, (
                    f"jump={abs(s_m - self._last.s_m):.1f}m > {max_jump:.1f}m"
                )

        px, py = self._point_at(s_m)
        plon, plat = self._to_wgs84.transform(px, py)
        result = MatchResult(
            s_m=float(s_m),
            cross_track_m=float(cross),
            seg_index=int(seg_idx),
            lat=float(plat),
            lon=float(plon),
            progress=float(s_m / self.total_length_m),
        )
        self._last = result
        self._last_time = now
        return result, "ok"

    # ------------------------------------------------------------------ #

    def _project(self, x: float, y: float, s_lo: float, s_hi: float) -> tuple[float, float, int]:
        """Chiếu điểm (x, y) lên các đoạn có s giao với [s_lo, s_hi].

        Trả (s_m, d_signed_m, seg_index) của đoạn gần nhất; d_signed > 0 nghĩa
        là điểm nằm bên PHẢI chiều đi của tuyến (quy ước panel, khớp control/ekf.py).
        """
        # Đoạn i chiếm [s_cum[i], s_cum[i+1]] — lấy các đoạn giao với cửa sổ.
        mask = (self._s_cum[1:] >= s_lo) & (self._s_cum[:-1] <= s_hi)
        idx = np.flatnonzero(mask)
        if idx.size == 0:  # cửa sổ rỗng (không xảy ra trong thực tế) -> toàn tuyến
            idx = np.arange(len(self._seg_len))

        p = np.array([x, y])
        rel = p - self._pts[idx]                                   # (M, 2)
        t = np.einsum("ij,ij->i", rel, self._seg_vec[idx]) / self._seg_len[idx] ** 2
        t = np.clip(t, 0.0, 1.0)
        closest = self._pts[idx] + t[:, None] * self._seg_vec[idx]  # (M, 2)
        dist = np.linalg.norm(p - closest, axis=1)

        best = int(np.argmin(dist))
        seg_i = int(idx[best])
        s_m = float(self._s_cum[seg_i] + t[best] * self._seg_len[seg_i])
        # Dấu: pháp tuyến phải của tiếp tuyến (tx, ty) là (ty, -tx).
        off = p - closest[best]
        tx, ty = self._seg_vec[seg_i] / self._seg_len[seg_i]
        d_signed = float(off[0] * ty - off[1] * tx)
        return s_m, d_signed, seg_i

    def _point_at(self, s_m: float) -> tuple[float, float]:
        """Toạ độ local (x, y) của điểm nằm tại s_m dọc tuyến."""
        s = float(np.clip(s_m, 0.0, self.total_length_m))
        seg_i = int(np.searchsorted(self._s_cum, s, side="right") - 1)
        seg_i = min(seg_i, len(self._seg_len) - 1)
        t = (s - self._s_cum[seg_i]) / self._seg_len[seg_i]
        pt = self._pts[seg_i] + t * self._seg_vec[seg_i]
        return float(pt[0]), float(pt[1])
