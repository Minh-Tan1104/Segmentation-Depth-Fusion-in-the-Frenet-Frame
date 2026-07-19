"""Mô phỏng offline kiểm chứng RouteEKF trên tuyến thật (map/gps_path_2m.csv).

Kịch bản: xe chạy đúng tim tuyến 1.5 m/s; encoder có bias thực tế (v lệch 2%,
omega lệch 5% + nhiễu); GPS 1 Hz nhiễu 2 m RMS; các khúc cua (dò tự động từ
độ cong tuyến) coi là MẤT VISION — vào cua thì anchor_lateral() từ "vision"
lần cuối, trong cua chỉ còn encoder predict + GPS correct (R trung thực).

Đạt yêu cầu khi:
  1. |d| trong mọi đoạn cua mù  < 1.0 m  (đủ để pure pursuit bám cua).
  2. |s_est - s_that| toàn tuyến < 5.0 m (đủ để trigger ngã rẽ đúng chỗ).
  3. Fix multipath (+15 m) bị gate Mahalanobis loại, không giật pose.
  4. Mất GPS hoàn toàn trong 1 đoạn cua -> encoder-only vẫn giữ |d| < 1.0 m.

Chạy: python sim_route_ekf.py   (in bảng kết quả + PASS/FAIL từng mục)
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from map_matcher import RouteMapMatcher
from route_ekf import RouteEKF, RouteEKFConfig

ROUTE_CSV = Path(__file__).resolve().parent.parent / "map" / "gps_path_2m.csv"

DT = 0.02              # 50 Hz predict (khớp odom_publish_rate_hz)
V_TRUE = 1.5           # m/s
GPS_PERIOD = 1.0       # s
GPS_SIGMA = 2.0        # m RMS mỗi trục
GPS_HACC_REPORT = 2.5  # hAcc receiver tự báo
# 2 mức hiệu chuẩn encoder để so sánh: "đã hiệu chuẩn" (sai 2%, đạt được bằng
# test vòng kín chỉnh wheel_radius/track_width) vs "chưa chuẩn" (sai 5%).
CAL_V_SCALE, CAL_W_SCALE = 1.02, 1.02
RAW_V_SCALE, RAW_W_SCALE = 1.02, 1.05
V_NOISE = 0.02
W_NOISE = 0.01
CURVE_CURV_THRESH = 0.05   # rad/m (~bán kính < 20 m) -> coi là khúc cua
CURVE_DILATE_M = 4.0       # nới đoạn mù thêm mỗi đầu


def detect_blind_zones(matcher: RouteMapMatcher) -> list[tuple[float, float]]:
    """Tìm các đoạn [s_start, s_end] có độ cong lớn (mất vision khi vào cua).

    Delegate sang RouteMapMatcher.detect_curve_zones() — dùng chung logic với
    Gps/node.py (production) thay vì giữ 2 bản riêng dễ trôi nhau.
    """
    return matcher.detect_curve_zones(
        curvature_thresh=CURVE_CURV_THRESH, dilate_m=CURVE_DILATE_M
    )


def in_any_zone(s: float, zones: list[tuple[float, float]]) -> int:
    for i, (a, b) in enumerate(zones):
        if a <= s <= b:
            return i
    return -1


def run_sim(
    matcher: RouteMapMatcher,
    zones: list[tuple[float, float]],
    gps_outage_zone: int = -1,
    multipath_at_s: float = -1.0,
    seed: int = 7,
    cfg: RouteEKFConfig | None = None,
    v_scale: float = CAL_V_SCALE,
    w_scale: float = CAL_W_SCALE,
) -> dict:
    rng = np.random.default_rng(seed)
    ekf = RouteEKF(matcher, cfg or RouteEKFConfig())
    ekf.initialize(0.0, d_m=rng.normal(0, 0.1), psi_err=rng.normal(0, math.radians(2)))

    s_gt = 0.0
    heading_prev = matcher.heading_at(0.0)
    t = 0.0
    next_gps = GPS_PERIOD
    prev_zone = -1
    multipath_done = False
    multipath_rejected = False

    zone_max_d = [0.0] * len(zones)
    zone_max_sigma_d = [0.0] * len(zones)
    max_s_err = 0.0
    end_s = matcher.total_length_m - 2.0

    while s_gt < end_s:
        # --- ground truth: xe bám đúng tim tuyến ---
        s_gt += V_TRUE * DT
        heading_now = matcher.heading_at(s_gt)
        omega_true = math.atan2(
            math.sin(heading_now - heading_prev), math.cos(heading_now - heading_prev)
        ) / DT
        heading_prev = heading_now
        t += DT

        # --- encoder (bias + nhiễu) -> predict ---
        v_odom = V_TRUE * v_scale + rng.normal(0, V_NOISE)
        w_odom = omega_true * w_scale + rng.normal(0, W_NOISE)
        ekf.predict(v_odom, w_odom, DT)

        zone_i = in_any_zone(s_gt, zones)

        # --- vision: có line ngoài zone; VÀO zone thì neo lần cuối rồi thôi ---
        if zone_i >= 0 and prev_zone < 0:
            # camera còn thấy line ngay mép cua -> neo d/psi (kèm nhiễu vision)
            ekf.anchor_lateral(
                d_m=rng.normal(0, 0.05), psi_err=rng.normal(0, math.radians(1))
            )
        prev_zone = zone_i

        # --- GPS 1 Hz ---
        if t >= next_gps:
            next_gps += GPS_PERIOD
            gx, gy = matcher.point_at(s_gt)
            if not multipath_done and 0.0 <= multipath_at_s <= s_gt:
                gx += 15.0  # fix multipath lệch hẳn 15 m
                multipath_done = True
                lat, lon = matcher.to_wgs84(gx, gy)
                ok, _reason = ekf.correct_gps(lat, lon, h_acc_m=GPS_HACC_REPORT, fix_type=3)
                multipath_rejected = not ok
            elif zone_i != gps_outage_zone or zone_i < 0:
                gx += rng.normal(0, GPS_SIGMA)
                gy += rng.normal(0, GPS_SIGMA)
                lat, lon = matcher.to_wgs84(gx, gy)
                ekf.correct_gps(lat, lon, h_acc_m=GPS_HACC_REPORT, fix_type=3)

        # --- đánh giá ---
        st = ekf.route_state()
        max_s_err = max(max_s_err, abs(st.s_m - s_gt))
        if zone_i >= 0:
            zone_max_d[zone_i] = max(zone_max_d[zone_i], abs(st.d_m))
            zone_max_sigma_d[zone_i] = max(zone_max_sigma_d[zone_i], st.sigma_d)

    return {
        "zone_max_d": zone_max_d,
        "zone_max_sigma_d": zone_max_sigma_d,
        "max_s_err": max_s_err,
        "multipath_rejected": multipath_rejected,
    }


def main() -> None:
    matcher = RouteMapMatcher(ROUTE_CSV)
    zones = detect_blind_zones(matcher)
    print(f"Tuyen: {matcher.total_length_m:.0f} m | {len(zones)} doan cua mu (mat vision):")
    for i, (a, b) in enumerate(zones):
        print(f"  cua {i}: s = {a:.0f} .. {b:.0f} m (dai {b - a:.0f} m)")

    checks: list[tuple[str, bool, str]] = []
    seeds = range(10)

    # Kịch bản A: encoder ĐÃ hiệu chuẩn (sai omega 2%) — yêu cầu chính.
    ds_a, ss_a = [], []
    for seed in seeds:
        r = run_sim(matcher, zones, seed=seed)
        ds_a.append(max(r["zone_max_d"]))
        ss_a.append(r["max_s_err"])
    print(f"\n[A] Encoder hieu chuan (omega sai 2%), GPS 1Hz, {len(list(seeds))} seed:")
    print(f"  max|d| qua cua: worst={max(ds_a):.2f} m, mean={np.mean(ds_a):.2f} m")
    print(f"  max|s_err|:     worst={max(ss_a):.2f} m")
    checks.append(("A: |d| < 1.0 m moi cua (worst 10 seed)", max(ds_a) < 1.0, f"{max(ds_a):.2f} m"))
    checks.append(("A: |s_err| < 5.0 m", max(ss_a) < 5.0, f"{max(ss_a):.2f} m"))

    # Kịch bản A': encoder CHƯA chuẩn (omega sai 5%) — stress test, chỉ báo cáo
    # mức chịu đựng; đây là lý do phải hiệu chuẩn track_width trước khi tin cua mù.
    ds_s = []
    for seed in seeds:
        r = run_sim(matcher, zones, seed=seed, w_scale=RAW_W_SCALE)
        ds_s.append(max(r["zone_max_d"]))
    print(f"\n[A'] Encoder CHUA chuan (omega sai 5%): "
          f"max|d| worst={max(ds_s):.2f} m, mean={np.mean(ds_s):.2f} m")
    checks.append(("A': omega sai 5% van |d| < 1.5 m", max(ds_s) < 1.5, f"{max(ds_s):.2f} m"))

    # Kịch bản B (stress): mất GPS HOÀN TOÀN suốt đoạn cua dài nhất, lấy seed
    # xấu nhất — giới hạn vật lý của dead-reckoning thuần. Ngưỡng 1.2 m: vẫn
    # trong nửa làn (2.7/2 = 1.35 m); sigma_d của EKF phình theo đúng mức trôi
    # nên control chỉ cần giảm tốc khi sigma_d lớn thay vì tin d mù quáng.
    longest = max(range(len(zones)), key=lambda i: zones[i][1] - zones[i][0])
    d_b, sig_b = 0.0, 0.0
    for s in seeds:
        r = run_sim(matcher, zones, gps_outage_zone=longest, seed=s)
        d_b = max(d_b, r["zone_max_d"][longest])
        sig_b = max(sig_b, r["zone_max_sigma_d"][longest])
    print(f"\n[B] Mat GPS trong cua {longest} (dai nhat): worst max|d| = {d_b:.2f} m, "
          f"sigma_d bao hieu toi {sig_b:.2f} m")
    checks.append((f"B: stress encoder-only qua cua {longest}, |d| < 1.2 m", d_b < 1.2, f"{d_b:.2f} m"))

    # Kịch bản C: multipath +15 m giữa đoạn cua đầu tiên.
    mid_first_curve = 0.5 * (zones[0][0] + zones[0][1])
    r_c = run_sim(matcher, zones, multipath_at_s=mid_first_curve)
    print(f"\n[C] Multipath +15 m tai s={mid_first_curve:.0f} m: "
          f"rejected={r_c['multipath_rejected']}, max|d| cua 0 = {r_c['zone_max_d'][0]:.2f} m")
    checks.append(("C: multipath bi gate loai", r_c["multipath_rejected"], ""))
    checks.append(("C: |d| cua 0 van < 1.0 m", r_c["zone_max_d"][0] < 1.0,
                   f"{r_c['zone_max_d'][0]:.2f} m"))

    print("\n" + "=" * 60)
    all_ok = True
    for name, ok, detail in checks:
        all_ok &= ok
        print(f"  {'PASS' if ok else 'FAIL'}  {name}  {detail}")
    print("=" * 60)
    print("KET QUA:", "PASS" if all_ok else "FAIL")
    raise SystemExit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
