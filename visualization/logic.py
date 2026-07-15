from __future__ import annotations

from typing import Any

import cv2
import numpy as np


class OverlayRenderer:
    """Vẽ overlay (làn đường, vật cản, panel Frenet, quỹ đạo tối ưu).

    Thuần cv2/numpy, không rclpy — `visualization/node.py` chỉ gọi `render()`
    rồi tự lo encode/publish ảnh kết quả.
    """

    def __init__(self, lane_width_m: float) -> None:
        self.lane_width_m = lane_width_m

    def render(
        self,
        frame: np.ndarray,
        detections: list[dict[str, Any]],
        frenet: dict[str, Any] | None,
    ) -> np.ndarray:
        # frame đã được perception_node blend sẵn segmentation — ở đây chỉ
        # vẽ thêm vị trí làn (bar dưới), panel Frenet, và box vật cản.
        overlay = frame.copy()

        self._draw_position_2d(overlay, frenet)
        if frenet is not None:
            self._draw_lane_on_image(overlay, frenet)
        # Panel Frenet luôn vẽ — xe ở giữa, không mất hẳn khi chưa thấy làn.
        self._draw_frenet_2d(overlay, frenet, detections)

        for detection in detections:
            bbox = detection["bbox"]
            x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
            label = detection["label"]
            confidence = detection["confidence"]
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 255), 2)
            text = f"{label} {confidence:.2f}"
            if detection["segmentation"]["enabled"]:
                text += f" | seg={detection['segmentation']['seg_ratio']:.2f}"
            cv2.putText(
                overlay,
                text,
                (x1, max(18, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )

        return overlay

    def _draw_position_2d(
        self, overlay: np.ndarray, frenet: dict[str, Any] | None
    ) -> None:
        """Lateral position meter — always drawn at bottom of frame."""
        h, w = overlay.shape[:2]
        bar_h  = 36
        bar_y0 = h - bar_h
        bar_y1 = h

        # Background bar
        cv2.rectangle(overlay, (0, bar_y0), (w, bar_y1), (25, 25, 25), -1)

        mid = w // 2
        half = w // 3          # display range ±w/3 pixels of d
        bar_mid_y = bar_y0 + bar_h // 2

        # Grid ticks at ±half/2 and ±half
        for tick_d in (-half, -half // 2, 0, half // 2, half):
            tx = mid + tick_d
            tick_h = 10 if tick_d == 0 else 6
            color  = (0, 180, 0) if tick_d == 0 else (80, 80, 80)
            lw     = 2 if tick_d == 0 else 1
            cv2.line(overlay, (tx, bar_mid_y - tick_h), (tx, bar_mid_y + tick_h), color, lw)

        if frenet is None:
            # No data — gray "?" centered
            cv2.putText(overlay, "no lane", (mid - 28, bar_mid_y + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 100, 100), 1)
            return

        d_f   = frenet.get("d_filtered", frenet["d_pixels"])
        hdg_f = self._heading_for_display(
            frenet.get("heading_px_filtered", frenet["heading_px_deg"])
        )

        # Clamp indicator to display range
        d_clamped = max(-half, min(half, d_f))
        line_x = mid + int(round(d_clamped))

        # Filled zone between vehicle center and lane line
        zone_x0 = min(mid, line_x)
        zone_x1 = max(mid, line_x)
        if zone_x1 > zone_x0:
            zone = overlay[bar_y0:bar_y1, zone_x0:zone_x1].copy()
            cv2.rectangle(zone, (0, 0), (zone_x1 - zone_x0, bar_h), (0, 80, 40), -1)
            cv2.addWeighted(zone, 0.4,
                            overlay[bar_y0:bar_y1, zone_x0:zone_x1], 0.6,
                            0, overlay[bar_y0:bar_y1, zone_x0:zone_x1])

        # Lane line tick (orange)
        cv2.line(overlay, (line_x, bar_y0 + 3), (line_x, bar_y1 - 3), (0, 140, 255), 3)

        # Vehicle marker (white triangle pointing down at bar center)
        tri = np.array([[mid, bar_y1 - 4],
                        [mid - 8, bar_y0 + 6],
                        [mid + 8, bar_y0 + 6]], np.int32)
        cv2.fillPoly(overlay, [tri], (230, 230, 230))

        # Heading arc above the lane line tick
        hdg_clamped = max(-60.0, min(60.0, hdg_f))
        arc_r = 14
        arc_x1 = line_x + int(arc_r * np.sin(np.radians(hdg_clamped)))
        arc_y1 = bar_y0 + 4 - int(arc_r * abs(np.cos(np.radians(hdg_clamped))))
        cv2.arrowedLine(overlay, (line_x, bar_y0 + 4), (arc_x1, arc_y1),
                        (0, 140, 255), 1, cv2.LINE_AA, tipLength=0.4)

        # Text: d and heading
        side = "R" if d_f > 0 else "L"
        label = f"d={abs(d_f):.0f}px {side}   hdg={hdg_f:+.1f}deg"
        tw = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)[0][0]
        cv2.putText(overlay, label,
                    (w - tw - 8, bar_mid_y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)

    def _draw_lane_on_image(
        self,
        overlay: np.ndarray,
        frenet: dict[str, Any],
    ) -> None:
        h, w = overlay.shape[:2]
        curve_pts = frenet["curve_pts"]

        # Draw multi-sample points on the frame
        for sx, sy in frenet.get("image_samples", []):
            cv2.circle(overlay, (int(round(sx)), int(round(sy))), 3, (80, 180, 255), -1)

        # Draw fitted reference line (green)
        for i in range(len(curve_pts) - 1):
            p1 = (int(round(curve_pts[i][0])), int(round(curve_pts[i][1])))
            p2 = (int(round(curve_pts[i + 1][0])), int(round(curve_pts[i + 1][1])))
            if 0 <= p1[0] < w and 0 <= p2[0] < w:
                cv2.line(overlay, p1, p2, (0, 255, 0), 2, cv2.LINE_AA)

        # Draw d offset arrow at reference y: vehicle center → line x
        veh_x = int(round(frenet["veh_x"]))
        line_x = int(round(frenet["line_x"]))
        y_ref = int(round(frenet["y_ref"]))
        y_ref = min(max(y_ref, 0), h - 1)
        cv2.arrowedLine(
            overlay,
            (veh_x, y_ref),
            (line_x, y_ref),
            (0, 165, 255),
            2,
            cv2.LINE_AA,
            tipLength=0.2,
        )
        cv2.circle(overlay, (veh_x, y_ref), 5, (0, 165, 255), -1)

        # raw values small label (top-left)
        d_raw   = frenet["d_pixels"]
        hdg_raw = self._heading_for_display(frenet["heading_px_deg"])
        side_r  = "R" if d_raw > 0 else "L"
        cv2.putText(overlay,
            f"raw d={abs(d_raw):.0f}px({side_r}) hdg_px={hdg_raw:+.1f}deg",
            (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 200, 150), 1, cv2.LINE_AA)

    @staticmethod
    def _heading_for_display(heading_deg: float) -> float:
        # Human-facing UI uses the opposite sign from the image-fit convention.
        return -float(heading_deg)

    def draw_frenet_panel(
        self,
        canvas: np.ndarray,
        frenet: dict[str, Any] | None,
        detections: list[dict[str, Any]],
        px0: int,
        py0: int,
        panel_w: int,
        panel_h: int,
        title: str = "Frenet",
    ) -> None:
        # frenet=None (chưa thấy làn) -> vẫn vẽ panel, chỉ thiếu dữ liệu làn.
        has_lane = frenet is not None
        frenet = frenet or {}

        h, w = canvas.shape[:2]
        panel_w = max(120, min(panel_w, w - px0 - 1))
        panel_h = max(140, min(panel_h, h - py0 - 1))

        lane_w_m    = float(frenet.get("lane_width_m", self.lane_width_m))
        half_lane_m = lane_w_m          # each side is full lane_width_m from center dash
        # kappa_ff: độ cong feed-forward hiện tại (control/node.py, panel: +=phải),
        # khác 0 CHỈ khi đang lái bằng GPS+encoder trong curve zone (xem
        # Gps/node.py). to_px() dưới đây uốn toàn bộ path/lưới theo cung tròn
        # bán kính 1/kappa_ff thay vì vẽ thẳng, để panel khớp hình dạng cua thật
        # đang lái — kappa_ff=0 thì mọi công thức rút gọn về y hệt trường hợp cũ.
        kappa = float(frenet.get("kappa_ff", 0.0))

        # ── Panel layout ──
        d_disp_m          = half_lane_m + 0.5   # ±(2.7+0.5) = ±3.2 m displayed
        t_top, t_bot      = 20, 32
        t_left, t_right   = 26, 6
        usable_w = panel_w - t_left - t_right
        usable_h = panel_h - t_top - t_bot

        ox = px0 + t_left + usable_w // 2   # panel x for d=0
        oy = py0 + t_top  + usable_h         # panel y for s=0 (bottom of usable area)

        s_max_m = max(float(frenet.get("s_max_m", 0.5)), 0.5)
        px_per_m_d = usable_w / (2.0 * d_disp_m)
        px_per_m_s = usable_h / s_max_m

        cx0, cx1 = px0 + t_left, px0 + panel_w - t_right
        cy0, cy1 = py0 + t_top,  oy

        def bend(d_val: float, s_val: float) -> tuple[float, float]:
            """Chiếu (d, s) Frenet-thẳng lên cung tròn bán kính 1/kappa (panel:
            +kappa = cua phải) — quy về (d_val, s_val) y hệt khi kappa=0."""
            if abs(kappa) < 1e-6:
                return d_val, s_val
            ks = kappa * s_val
            c, sn = np.cos(ks), np.sin(ks)
            return (1.0 - c) / kappa + d_val * c, sn / kappa - d_val * sn

        def to_px(d_val: float, s_val: float) -> tuple[int, int]:
            bd, bs = bend(d_val, s_val)
            return (
                ox + int(round(bd * px_per_m_d)),
                oy - int(round(bs * px_per_m_s)),
            )

        # Background + border — viền cam dày khi đang lái bằng GPS+encoder
        # (using_gps_route, control/node.py:_use_gps_route) để nhận ra ngay
        # cả khi liếc thoáng qua, không cần đọc số.
        using_gps = bool(frenet.get("using_gps_route", False))
        border_color = (0, 140, 255) if using_gps else (120, 120, 120)
        border_w = 3 if using_gps else 1
        cv2.rectangle(canvas, (px0, py0), (px0 + panel_w, py0 + panel_h),
                      (255, 255, 255), -1)
        cv2.rectangle(canvas, (px0, py0), (px0 + panel_w, py0 + panel_h),
                      border_color, border_w)
        cv2.putText(canvas, title,
                    (px0 + 4, py0 + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.30, (70, 70, 70), 1)
        if using_gps:
            banner = "GPS + ENCODER"
            (tw, th), _ = cv2.getTextSize(banner, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 2)
            bx0, by0 = px0 + panel_w - tw - 14, py0 + 2
            cv2.rectangle(canvas, (bx0, by0), (bx0 + tw + 10, by0 + th + 8),
                          (0, 140, 255), -1)
            cv2.putText(canvas, banner, (bx0 + 5, by0 + th + 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 2, cv2.LINE_AA)

        # ── S-axis grid (ticks every 1 m — tilted khi kappa != 0) ──
        s_t = 1.0
        while s_t <= s_max_m + 0.01:
            p_lo = to_px(-d_disp_m, s_t)
            p_hi = to_px(d_disp_m, s_t)
            _, ty_mid = to_px(0.0, s_t)
            if cy0 <= ty_mid <= cy1:
                cv2.line(canvas, p_lo, p_hi, (210, 210, 210), 1)
                cv2.putText(canvas, f"{s_t:.0f}m",
                            (px0 + 1, ty_mid + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.24, (90, 90, 90), 1)
            s_t += 1.0

        # ── Lane boundaries (đường cong offset ±half_lane_m, vẽ đứt nét theo s) ──
        s_step = max(0.05, s_max_m / 80.0)
        for sign in (-1.0, 1.0):
            d_bound = sign * half_lane_m
            s_cur, dash, seg = 0.0, True, []
            while s_cur <= s_max_m + s_step:
                if dash:
                    seg.append(to_px(d_bound, s_cur))
                elif seg:
                    if len(seg) >= 2:
                        cv2.polylines(canvas, [np.array(seg, dtype=np.int32)],
                                      False, (100, 60, 60), 1, cv2.LINE_AA)
                    seg = []
                if s_cur % (s_step * 3) < s_step:
                    dash = not dash
                s_cur += s_step
            if len(seg) >= 2:
                cv2.polylines(canvas, [np.array(seg, dtype=np.int32)],
                              False, (100, 60, 60), 1, cv2.LINE_AA)
            lx, ly = to_px(d_bound, 0.0)
            label = f"{d_bound:+.2f}m"
            lbl_x = lx - 26 if sign < 0 else lx + 2
            cv2.putText(canvas, label, (lbl_x, cy0 + 9),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.24, (130, 70, 70), 1)

        # ── Reference path (d=0, cong theo kappa nếu đang GPS feed-forward) ──
        ref_pts = [to_px(0.0, s) for s in np.linspace(0.0, s_max_m, 24)]
        cv2.polylines(canvas, [np.array(ref_pts, dtype=np.int32)],
                      False, (0, 180, 0), 1, cv2.LINE_AA)
        cv2.putText(canvas, "s(m)", (ox + 2, cy0 + 9),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0, 160, 0), 1)

        # ── Sample points class PHỤ (vd curb, viz-only) — vẽ TRƯỚC để điểm
        # line target đè lên nếu trùng; màu xanh dương khớp màu curb trên ảnh
        # blend (_SEG_COLORS[0], xem perception/logic.py) ──
        for d_f_m, s_f_m in frenet.get("sample_frenet_m_other", []):
            pp = to_px(d_f_m, s_f_m)
            if cx0 <= pp[0] <= cx1 and cy0 <= pp[1] <= cy1:
                cv2.circle(canvas, pp, 2, (255, 140, 90), -1)

        # ── Sample points (blue dots, metric positions) ──
        for d_f_m, s_f_m in frenet.get("sample_frenet_m", []):
            pp = to_px(d_f_m, s_f_m)
            if cx0 <= pp[0] <= cx1 and cy0 <= pp[1] <= cy1:
                cv2.circle(canvas, pp, 2, (80, 180, 255), -1)

        # ── Dash outlines (orange rectangles, metric positions) ──
        for d_f_m, s_f_m in frenet.get("dash_frenet_m", []):
            pp = to_px(d_f_m, s_f_m)
            if cx0 <= pp[0] <= cx1 and cy0 <= pp[1] <= cy1:
                cv2.rectangle(canvas,
                              (pp[0] - 6, pp[1] - 4), (pp[0] + 6, pp[1] + 4),
                              (220, 140, 60), 1)

        # ── Detected obstacles projected into Frenet (red markers) ──
        for detection in detections:
            fr_det = detection.get("frenet")
            if not fr_det or not fr_det.get("available"):
                continue
            d_plot = float(fr_det.get("d_m_filtered", fr_det["d_m"]))
            s_plot = float(fr_det.get("s_m_filtered", fr_det["s_m"]))
            pp = to_px(d_plot, s_plot)
            if cx0 <= pp[0] <= cx1 and cy0 <= pp[1] <= cy1:
                cv2.circle(canvas, pp, 5, (40, 40, 220), -1)
                cv2.circle(canvas, pp, 7, (120, 120, 255), 1)
                label = detection["label"][:3].upper()
                cv2.putText(
                    canvas,
                    label,
                    (pp[0] + 8, max(cy0 + 10, pp[1] - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.28,
                    (180, 180, 255),
                    1,
                    cv2.LINE_AA,
                )

        # ── Candidate paths considered by the planner (dim gray polylines) ──
        for cand_path in frenet.get("candidate_paths", []):
            pts = []
            for d_val, s_val in cand_path:
                pp = to_px(d_val, s_val)
                if cx0 <= pp[0] <= cx1 and cy0 <= pp[1] <= cy1:
                    pts.append(pp)
            if len(pts) >= 2:
                cv2.polylines(canvas, [np.array(pts, dtype=np.int32)],
                              False, (90, 90, 90), 1, cv2.LINE_AA)

        # ── Frenet optimal path (cyan polyline) ──
        opt_path = frenet.get("optimal_path")
        if opt_path:
            pts = []
            for d_val, s_val in opt_path:
                pp = to_px(d_val, s_val)
                if cx0 <= pp[0] <= cx1 and cy0 <= pp[1] <= cy1:
                    pts.append(pp)
            if len(pts) >= 2:
                cv2.polylines(canvas, [np.array(pts, dtype=np.int32)],
                              False, (255, 255, 0), 2, cv2.LINE_AA)
                cv2.circle(canvas, pts[-1], 3, (255, 255, 0), -1)

        # ── Điểm lookahead pure pursuit đang nhắm tới (tâm hồng + vòng viền) ──
        lookahead = frenet.get("lookahead_point")
        if lookahead:
            d_la, s_la = float(lookahead[0]), float(lookahead[1])
            pp = to_px(d_la, s_la)
            if cx0 <= pp[0] <= cx1 and cy0 <= pp[1] <= cy1:
                cv2.circle(canvas, pp, 6, (220, 0, 220), -1)
                cv2.circle(canvas, pp, 9, (255, 180, 255), 2, cv2.LINE_AA)
                cv2.putText(canvas, "lookahead", (pp[0] + 10, pp[1] + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.28, (220, 0, 220), 1, cv2.LINE_AA)

        # ── Vehicle rectangle (xoay theo heading) ──
        d_filt_m = float(frenet.get("d_meters_filtered", frenet.get("d_meters", 0.0)))
        # d > 0 → reference line is RIGHT of vehicle → vehicle is LEFT of reference
        # Negate so vehicle appears on the correct side of the green line
        vx, _    = to_px(-d_filt_m, 0.0)
        vx       = int(np.clip(vx, cx0 + 7, cx1 - 7))
        vy       = cy1

        hdg_filt = self._heading_for_display(
            float(frenet.get("heading_filtered", frenet.get("heading_deg", 0.0)))
        )
        hdg_rad  = np.radians(hdg_filt)
        fwd      = np.array([np.sin(hdg_rad), -np.cos(hdg_rad)])   # hướng mũi xe
        right    = np.array([np.cos(hdg_rad), np.sin(hdg_rad)])    # hướng ngang xe
        center   = np.array([float(vx), float(vy) - 6.0])
        half_len, half_wid = 6.0, 5.0
        corners = np.array(
            [
                center + fwd * half_len + right * half_wid,   # mũi-phải
                center + fwd * half_len - right * half_wid,   # mũi-trái
                center - fwd * half_len - right * half_wid,   # đuôi-trái
                center - fwd * half_len + right * half_wid,   # đuôi-phải
            ],
            dtype=np.int32,
        )
        cv2.fillPoly(canvas, [corners], (0, 140, 255))
        cv2.polylines(canvas, [corners], True, (0, 90, 170), 1, cv2.LINE_AA)

        # ── Heading arrow từ mũi xe ──
        alen   = int(min(28, usable_h * 0.22))
        nose   = center + fwd * half_len
        tip    = center + fwd * (half_len + alen)
        nose_pt = (int(np.clip(nose[0], cx0, cx1)), int(np.clip(nose[1], cy0, cy1)))
        tip_pt  = (int(np.clip(tip[0], cx0, cx1)), int(np.clip(tip[1], cy0, cy1)))
        cv2.arrowedLine(canvas, nose_pt, tip_pt,
                        (0, 165, 255), 1, cv2.LINE_AA, tipLength=0.3)

        # ── d arrow + value label on map ──
        if abs(d_filt_m) > 0.02:
            arr_y = cy1 + 10
            cv2.arrowedLine(canvas, (vx, arr_y), (ox, arr_y),
                            (0, 165, 255), 1, cv2.LINE_AA, tipLength=0.2)
            side = "R" if d_filt_m > 0 else "L"
            d_label = f"{abs(d_filt_m):.2f}m{side}"
            lx = (vx + ox) // 2
            cv2.putText(canvas, d_label, (lx - 14, arr_y - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.28, (0, 200, 255), 1)

        # ── Bottom metric text — matches position bar convention: d>0 = line is RIGHT ──
        if has_lane:
            side     = "R" if d_filt_m > 0 else "L"
            hdg_show = self._heading_for_display(
                float(frenet.get("heading_filtered", frenet.get("heading_deg", 0.0)))
            )
            text = f"d={abs(d_filt_m):.2f}m {side} {hdg_show:+.0f}deg"
            color = (60, 60, 60)
        else:
            text = "no lane"
            color = (140, 140, 140)
        if abs(kappa) > 1e-6:
            text += f"   GPS+encoder R={abs(1.0/kappa):.1f}m ({'phai' if kappa > 0 else 'trai'})"
            color = (0, 120, 200)
        cv2.putText(canvas, text,
                    (px0 + 4, py0 + panel_h - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.26, color, 1)

    def draw_encoder_panel(
        self,
        canvas: np.ndarray,
        odom_trail: list[tuple[float, float]],
        ekf_state: list[float] | None,
        px0: int,
        py0: int,
        panel_w: int,
        panel_h: int,
    ) -> None:
        """Quỹ đạo (x, y) tích phân từ /odom (gốc tọa độ = điểm xuất phát,
        encoder_node luôn khởi tạo x=y=0) + số liệu EKF (s, d, psi, v) hiện
        tại — debug xác nhận encoder/EKF còn cập nhật khi mất line."""
        h, w = canvas.shape[:2]
        panel_w = max(120, min(panel_w, w - px0 - 1))
        panel_h = max(140, min(panel_h, h - py0 - 1))

        cv2.rectangle(canvas, (px0, py0), (px0 + panel_w, py0 + panel_h),
                      (255, 255, 255), -1)
        cv2.rectangle(canvas, (px0, py0), (px0 + panel_w, py0 + panel_h),
                      (120, 120, 120), 1)
        cv2.putText(canvas, "Encoder odom",
                    (px0 + 4, py0 + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.30, (70, 70, 70), 1)

        t_top, t_bot = 20, 36
        t_left, t_right = 6, 6
        cx0, cx1 = px0 + t_left, px0 + panel_w - t_right
        cy0, cy1 = py0 + t_top, py0 + panel_h - t_bot
        usable_w = max(1, cx1 - cx0)
        usable_h = max(1, cy1 - cy0)
        ox = (cx0 + cx1) // 2
        oy = (cy0 + cy1) // 2

        # Auto-scale theo bounding box của trail, tối thiểu ±1m để không
        # chia 0 lúc xe gần như chưa di chuyển.
        span_m = 1.0
        for x_m, y_m in odom_trail:
            span_m = max(span_m, abs(x_m), abs(y_m))
        span_m *= 1.2
        px_per_m = min(usable_w, usable_h) / (2.0 * span_m)

        def to_px(x_m: float, y_m: float) -> tuple[int, int]:
            return (
                ox + int(round(x_m * px_per_m)),
                oy - int(round(y_m * px_per_m)),
            )

        # Lưới + gốc tọa độ (điểm bám — vị trí encoder_node khởi tạo)
        cv2.line(canvas, (cx0, oy), (cx1, oy), (210, 210, 210), 1)
        cv2.line(canvas, (ox, cy0), (ox, cy1), (210, 210, 210), 1)
        cv2.circle(canvas, (ox, oy), 4, (0, 160, 0), 1)
        cv2.putText(canvas, "origin", (ox + 6, oy - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.24, (0, 130, 0), 1)

        # Trail (đường đi thực tế theo /odom)
        pts = [to_px(x_m, y_m) for x_m, y_m in odom_trail]
        if len(pts) >= 2:
            cv2.polylines(canvas, [np.array(pts, dtype=np.int32)],
                          False, (220, 140, 60), 2, cv2.LINE_AA)
        if pts:
            cv2.circle(canvas, pts[-1], 5, (40, 40, 220), -1)
            cv2.circle(canvas, pts[-1], 7, (120, 120, 255), 1)

        # Số liệu EKF — số đổi liên tục == predict()/correct() vẫn chạy.
        if ekf_state is not None and len(ekf_state) >= 4:
            s_v, d_v, psi_v, v_v = ekf_state[:4]
            line1 = f"EKF s={s_v:.2f} d={d_v:.2f}m"
            line2 = f"psi={np.degrees(psi_v):+.1f}deg v={v_v:.2f}m/s"
            color = (60, 60, 60)
        else:
            line1, line2 = "EKF: no data yet", ""
            color = (140, 140, 140)
        cv2.putText(canvas, line1, (px0 + 4, py0 + panel_h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.26, color, 1)
        if line2:
            cv2.putText(canvas, line2, (px0 + 4, py0 + panel_h - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.26, color, 1)

    def draw_gps_panel(
        self,
        canvas: np.ndarray,
        route_xy: np.ndarray | None,
        car_xy: tuple[float, float] | None,
        car_heading: float | None,
        sigma_d: float | None,
        px0: int,
        py0: int,
        panel_w: int,
        panel_h: int,
        curve_zones: list[tuple[float, float]] | None = None,
    ) -> None:
        """Vẽ tuyến ghi sẵn (map/gps_path_2m.csv, frame mét cục bộ) + vị trí xe
        hiện tại chiếu từ /gps/route_state (Gps/node.py). route_xy=None nghĩa
        là chưa nạp được tuyến; car_xy=None nghĩa là RouteEKF chưa initialize
        (chưa có fix GPS đầu tiên) — vẫn vẽ khung panel, chỉ báo trạng thái.

        curve_zones: đoạn [s_start, s_end] (RouteMapMatcher.detect_curve_zones())
        được tô màu cam riêng — đúng đoạn mà control_node bật curvature
        feed-forward khi chạy GPS assist (xem Gps/node.py: _curve_ff_kappa)."""
        h, w = canvas.shape[:2]
        panel_w = max(120, min(panel_w, w - px0 - 1))
        panel_h = max(140, min(panel_h, h - py0 - 1))

        cv2.rectangle(canvas, (px0, py0), (px0 + panel_w, py0 + panel_h),
                      (255, 255, 255), -1)
        cv2.rectangle(canvas, (px0, py0), (px0 + panel_w, py0 + panel_h),
                      (120, 120, 120), 1)
        cv2.putText(canvas, "GPS route",
                    (px0 + 4, py0 + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.30, (70, 70, 70), 1)

        if route_xy is None or len(route_xy) < 2:
            cv2.putText(canvas, "route CSV not loaded",
                        (px0 + 4, py0 + panel_h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, (140, 140, 140), 1)
            return

        t_top, t_bot = 20, 16
        t_left, t_right = 6, 6
        cx0, cx1 = px0 + t_left, px0 + panel_w - t_right
        cy0, cy1 = py0 + t_top, py0 + panel_h - t_bot
        usable_w = max(1, cx1 - cx0)
        usable_h = max(1, cy1 - cy0)

        x_min, y_min = route_xy.min(axis=0)
        x_max, y_max = route_xy.max(axis=0)
        span_x = max(1.0, (x_max - x_min) * 1.1)
        span_y = max(1.0, (y_max - y_min) * 1.1)
        px_per_m = min(usable_w / span_x, usable_h / span_y)
        cx_m = (x_min + x_max) / 2.0
        cy_m = (y_min + y_max) / 2.0
        ox = (cx0 + cx1) // 2
        oy = (cy0 + cy1) // 2

        def to_px(x_m: float, y_m: float) -> tuple[int, int]:
            return (
                ox + int(round((x_m - cx_m) * px_per_m)),
                oy - int(round((y_m - cy_m) * px_per_m)),
            )

        pts = [to_px(x, y) for x, y in route_xy]
        cv2.polylines(canvas, [np.array(pts, dtype=np.int32)],
                      False, (160, 40, 130), 2, cv2.LINE_AA)

        if curve_zones:
            # s dọc route_xy tính lại từ khoảng cách các đỉnh polyline — cùng
            # điểm/thứ tự mà RouteMapMatcher dùng để tính s nội bộ nên khớp
            # đúng biên (s_start, s_end) của curve_zones.
            seg_len = np.linalg.norm(np.diff(route_xy, axis=0), axis=1)
            s_cum = np.concatenate([[0.0], np.cumsum(seg_len)])
            for s0, s1 in curve_zones:
                idx = np.flatnonzero((s_cum >= s0) & (s_cum <= s1))
                if len(idx) < 2:
                    continue
                zone_pts = [pts[i] for i in idx]
                cv2.polylines(canvas, [np.array(zone_pts, dtype=np.int32)],
                              False, (0, 140, 255), 3, cv2.LINE_AA)

        if car_xy is None:
            cv2.putText(canvas, "cho fix GPS dau tien...",
                        (px0 + 4, py0 + panel_h - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.28, (140, 140, 140), 1)
            return

        car_px = to_px(*car_xy)
        # Màu marker theo độ tin cậy sigma_d — xanh (tin được) -> đỏ (đang trôi).
        if sigma_d is None:
            marker_color = (120, 120, 120)
        else:
            t = float(np.clip(sigma_d / 1.0, 0.0, 1.0))  # 0m xanh, >=1m đỏ
            marker_color = (0, int(round(200 * (1 - t))), int(round(220 * t)))
        cv2.circle(canvas, car_px, 5, marker_color, -1)
        cv2.circle(canvas, car_px, 7, (60, 60, 60), 1)
        if car_heading is not None:
            tip = (
                car_px[0] + int(round(10 * np.cos(car_heading))),
                car_px[1] - int(round(10 * np.sin(car_heading))),
            )
            cv2.arrowedLine(canvas, car_px, tip, (60, 60, 60), 1, tipLength=0.4)

        sigma_txt = "n/a" if sigma_d is None else f"{sigma_d:.2f}m"
        cv2.putText(canvas, f"sigma_d={sigma_txt}",
                    (px0 + 4, py0 + panel_h - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.26, (60, 60, 60), 1)

    def _draw_frenet_2d(
        self,
        overlay: np.ndarray,
        frenet: dict[str, Any] | None,
        detections: list[dict[str, Any]],
        panel_w: int = 175,
        panel_h: int = 220,
    ) -> None:
        h, w = overlay.shape[:2]

        # Position above the 36-px position bar
        bar_h = 36
        mx, my = 10, 10
        px0 = w - panel_w - mx
        py0 = h - bar_h - panel_h - my
        self.draw_frenet_panel(
            overlay, frenet, detections, px0, py0, panel_w, panel_h
        )
