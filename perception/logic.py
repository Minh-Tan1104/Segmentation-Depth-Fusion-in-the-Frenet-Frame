from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np
import torch
from scipy.interpolate import UnivariateSpline
from ultralytics import YOLO


@dataclass
class PerceptionConfig:
    yolo_model_path: str
    seg_model_path: str
    device: str = "auto"
    yolo_conf: float = 0.35
    yolo_iou: float = 0.45
    seg_conf: float = 0.35
    seg_x_min: float = 0.25
    seg_x_max: float = 0.75
    seg_min_area: float = 500.0
    seg_target_class_id: int = 1
    cam_cx: float = -1.0
    seg_y_step: int = 4
    frenet_alpha: float = 0.3
    lane_width_m: float = 2.7
    depth_scale: float = 0.001
    segmentation_alpha: float = 0.35
    obstacle_alpha: float = 0.05
    obstacle_track_ttl: int = 5
    obstacle_match_px: float = 90.0
    obstacle_match_s: float = 2.0
    obstacle_match_d: float = 1.2
    obstacle_match_cost_max: float = 1.35
    obstacle_depth_percentile: float = 35.0
    detection_conf_gate: float = 0.45
    # Chạy seg cả 2 class (curb + dashed_yellow_line) thay vì chỉ target —
    # class phụ CHỈ đi vào nhánh hiển thị (blend ảnh + chấm trên panel Frenet
    # 2D), KHÔNG tham gia fit d/heading, không vào seg_mask mà detection dùng
    # tính seg_ratio. Tắt (false) = hành vi cũ y hệt (chỉ 1 class).
    seg_viz_all_classes: bool = True


@dataclass
class FrameResult:
    visual_frame: np.ndarray
    segmentation_meta: dict[str, Any]
    detections: list[dict[str, Any]]
    frenet_viz: dict[str, Any] | None
    d_meters_filtered: float | None
    heading_filtered: float | None
    coeffs: list[float] = field(default_factory=list)
    # Timing 2 lần YOLO tách riêng [ms] — node.py ghi ra CSV khi log_timing_enable.
    # Đo bằng perf_counter quanh self.seg.predict / self.yolo.predict.
    seg_ms: float = 0.0
    det_ms: float = 0.0


class PerceptionLogic:
    """Seg làn -> Frenet fit -> detection vật cản -> chiếu vào Frenet.

    Không có rclpy/pub-sub ở đây — `perception/node.py` chỉ gọi
    `set_depth_image`/`set_camera_intrinsics`/`process_frame` rồi tự lo publish.
    """

    _YOLO_CLASSES = [0, 1, 2, 3, 5, 7]
    # BGR colors — one per class id (0: curb, 1: dashed_yellow_line)
    _SEG_COLORS = [
        (255, 80, 80),
        (80, 255, 80),
    ]

    def __init__(self, config: PerceptionConfig, logger: Any | None = None) -> None:
        self.cfg = config
        self._logger = logger

        self._depth_img: np.ndarray | None = None
        self._cam_intr: dict[str, float] | None = None
        self.frame_count = 0
        self._obstacle_tracks: dict[int, dict[str, Any]] = {}
        self._next_obstacle_track_id = 1
        self._d_ema: float | None = None
        self._hdg_px_ema: float | None = None
        self._hdg_raw: float | None = None
        self._d_m_raw: float | None = None  # giá trị đo gần nhất (m), không EMA

        self.device_name = self._resolve_device(config.device.strip().lower())

        self.yolo = YOLO(config.yolo_model_path)
        if getattr(self.yolo, "task", None) != "detect":
            raise RuntimeError(
                f"Model '{config.yolo_model_path}' is not a YOLO detection model."
            )
        if isinstance(self.yolo.names, dict):
            class_items = self.yolo.names.items()
        else:
            class_items = enumerate(self.yolo.names)
        self.class_names = {int(cid): str(lbl) for cid, lbl in class_items}
        self._warm_up_yolo()

        self.seg = YOLO(config.seg_model_path)
        if getattr(self.seg, "task", None) != "segment":
            raise RuntimeError(
                f"Model '{config.seg_model_path}' is not a YOLO segmentation model."
            )
        self._warm_up_seg()

    # ── trạng thái đầu vào (depth/camera_info) ───────────────────────────
    @property
    def has_depth(self) -> bool:
        return self._depth_img is not None

    @property
    def has_camera_info(self) -> bool:
        return self._cam_intr is not None

    def set_depth_image(self, raw: np.ndarray, is_integer_encoding: bool) -> None:
        depth = raw.astype(np.float32, copy=False)
        if is_integer_encoding:
            depth = depth * self.cfg.depth_scale
        self._depth_img = depth

    def set_camera_intrinsics(self, fx: float, fy: float, cx: float, cy: float) -> None:
        if self._cam_intr is not None:
            return
        self._cam_intr = {"fx": fx, "fy": fy, "cx": cx, "cy": cy}

    def _resolve_device(self, requested: str) -> str:
        if requested in {"", "auto"}:
            return "cuda:0" if torch.cuda.is_available() else "cpu"
        if requested.startswith("cuda") and not torch.cuda.is_available():
            if self._logger is not None:
                self._logger.warning(
                    "CUDA was requested but is not available, falling back to CPU."
                )
            return "cpu"
        return requested

    def _warm_up_yolo(self) -> None:
        dummy = np.zeros((32, 32, 3), dtype=np.uint8)
        self.yolo.predict(
            source=dummy,
            verbose=False,
            conf=self.cfg.yolo_conf,
            iou=self.cfg.yolo_iou,
            classes=self._YOLO_CLASSES,
            device=self.device_name,
        )

    def _seg_classes_filter(self) -> list[int] | None:
        """None = giữ mọi class (viz cả 2), [target] = như cũ. NMS của
        ultralytics chạy per-class (agnostic=False) nên bỏ filter KHÔNG đổi
        kết quả của class target — class kia chỉ thêm vào, không đè."""
        return None if self.cfg.seg_viz_all_classes else [self.cfg.seg_target_class_id]

    def _warm_up_seg(self) -> None:
        dummy = np.zeros((320, 640, 3), dtype=np.uint8)
        self.seg.predict(
            source=dummy,
            verbose=False,
            conf=self.cfg.seg_conf,
            iou=self.cfg.yolo_iou,
            classes=self._seg_classes_filter(),
            device=self.device_name,
        )

    # ── xử lý 1 frame ─────────────────────────────────────────────────────
    def process_frame(self, bgr: np.ndarray, frame_index: int) -> FrameResult:
        """Caller phải đảm bảo has_depth/has_camera_info=True trước khi gọi."""
        self.frame_count = frame_index
        depth_img = self._depth_img
        cam_intr = self._cam_intr

        # perf_counter quanh 2 lần YOLO (seg + det) để đo tải inference THẬT
        # tách riêng — GPU nên bao gồm cả kernel launch + copy, đúng cái xe chịu.
        _t_seg0 = time.perf_counter()
        seg_mask, seg_mask_viz, seg_color, segmentation_meta, frenet = (
            self._run_yolo_segmentation(bgr, depth_img, cam_intr)
        )
        seg_ms = (time.perf_counter() - _t_seg0) * 1000.0
        # Không có làn không có nghĩa là không có gì để publish — ảnh
        # (visual_frame) và detection vật cản vẫn hợp lệ độc lập với làn.
        # Chỉ phần Frenet/lane sẽ là None.
        self._update_frenet(frenet)
        # detection dùng seg_mask (target-only) — seg_ratio không đổi khi bật
        # seg_viz_all_classes; blend ảnh dùng seg_mask_viz (cả class phụ).
        _t_det0 = time.perf_counter()
        detections = self._run_yolo_detection(bgr, seg_mask, depth_img, cam_intr, frenet)
        det_ms = (time.perf_counter() - _t_det0) * 1000.0
        visual_frame = self._build_visual_frame(bgr, seg_mask_viz, seg_color)
        frenet_viz = self._build_frenet_viz(frenet) if frenet is not None else None

        return FrameResult(
            visual_frame=visual_frame,
            segmentation_meta=segmentation_meta,
            detections=detections,
            frenet_viz=frenet_viz,
            d_meters_filtered=self._d_m_raw,
            heading_filtered=self._hdg_raw,
            coeffs=list(frenet.get("coeffs", [])) if frenet is not None else [],
            seg_ms=seg_ms,
            det_ms=det_ms,
        )

    def _run_yolo_segmentation(
        self,
        bgr: np.ndarray,
        depth_img: np.ndarray,
        cam_intr: dict[str, float],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any], dict[str, Any] | None]:
        image_h, image_w = bgr.shape[:2]
        seg_mask = np.zeros((image_h, image_w), dtype=np.uint8)   # CHỈ class target — detection dùng tính seg_ratio
        seg_mask_viz = np.zeros((image_h, image_w), dtype=np.uint8)  # target + class phụ — chỉ để blend ảnh
        seg_color = np.zeros((image_h, image_w, 3), dtype=np.uint8)

        result = self.seg.predict(
            source=bgr,
            verbose=False,
            conf=self.cfg.seg_conf,
            iou=self.cfg.yolo_iou,
            classes=self._seg_classes_filter(),
            device=self.device_name,
        )[0]

        if result.masks is None or len(result.masks) == 0:
            return seg_mask, seg_mask_viz, seg_color, {
                "mode": "yolov8seg",
                "instance_count": 0,
                "mask_pixels": 0,
            }, None

        classes = (
            result.boxes.cls.cpu().numpy().astype(int)
            if result.boxes is not None
            else np.zeros(len(result.masks.xy), dtype=int)
        )

        x_lo = self.cfg.seg_x_min * image_w
        x_hi = self.cfg.seg_x_max * image_w

        accepted: list[np.ndarray] = []       # class target — fit d/heading như cũ
        other_polys: list[np.ndarray] = []    # class phụ — CHỈ hiển thị
        for polygon, class_id in zip(result.masks.xy, classes):
            if len(polygon) < 3:
                continue
            pts = np.array(polygon, dtype=np.int32)
            if cv2.contourArea(pts) < self.cfg.seg_min_area:
                continue
            centroid_x = float(np.mean(polygon[:, 0]))
            if not (x_lo <= centroid_x <= x_hi):
                continue
            color = self._SEG_COLORS[int(class_id) % len(self._SEG_COLORS)]
            cv2.fillPoly(seg_mask_viz, [pts], 255)
            cv2.fillPoly(seg_color, [pts], color)
            if int(class_id) == self.cfg.seg_target_class_id:
                accepted.append(polygon.astype(np.float32))
                cv2.fillPoly(seg_mask, [pts], 255)  # mask cho detection: target-only
            else:
                other_polys.append(polygon.astype(np.float32))

        frenet = self._fit_line_frenet(
            accepted, image_h, image_w, depth_img, cam_intr, other_polys=other_polys
        )

        meta: dict[str, Any] = {
            "mode": "yolov8seg",
            "instance_count": len(accepted),
            "other_instance_count": len(other_polys),
            "mask_pixels": int(np.count_nonzero(seg_mask)),
        }
        if frenet is not None:
            meta["frenet"] = {
                "fit_mode": frenet["fit_mode"],
                "d_pixels": frenet["d_pixels"],
                "heading_px_deg": frenet["heading_px_deg"],
                "n_dashes": frenet["n_dashes"],
                "degree": frenet["degree"],
                "d_meters": frenet["d_meters"],
                "heading_deg": frenet["heading_deg"],
                "spline_order_metric": frenet.get("spline_order_metric", 1),
                "depth_points": frenet.get("depth_points", 0),
            }

        return seg_mask, seg_mask_viz, seg_color, meta, frenet

    @staticmethod
    def _sample_dash_points(
        polygon: np.ndarray, y_step: int
    ) -> list[tuple[float, float]]:
        """Rasterize one dash polygon and return (x, y) samples every y_step rows."""
        pts = polygon.astype(np.int32)
        y_min, y_max = int(pts[:, 1].min()), int(pts[:, 1].max())
        x_min, x_max = int(pts[:, 0].min()), int(pts[:, 0].max())
        if y_max <= y_min or x_max <= x_min:
            return [(float(np.mean(polygon[:, 0])), float(np.mean(polygon[:, 1])))]
        h, w = y_max - y_min + 1, x_max - x_min + 1
        mask = np.zeros((h, w), dtype=np.uint8)
        shifted = pts.copy()
        shifted[:, 0] -= x_min
        shifted[:, 1] -= y_min
        cv2.fillPoly(mask, [shifted], 255)
        samples: list[tuple[float, float]] = []
        for y_local in range(0, h, y_step):
            row_xs = np.where(mask[y_local] > 0)[0]
            if len(row_xs):
                x_mid = (float(row_xs[0]) + float(row_xs[-1])) / 2.0
                samples.append((x_mid + x_min, float(y_local + y_min)))
        return samples or [(float(np.mean(polygon[:, 0])), float(np.mean(polygon[:, 1])))]

    @staticmethod
    def _collapse_axis_samples(
        indep_samples: np.ndarray,
        dep_samples: np.ndarray,
        bin_size: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        bins = np.round(indep_samples / bin_size).astype(np.int64)
        unique_bins, inverse = np.unique(bins, return_inverse=True)
        indep_sum = np.zeros(len(unique_bins), dtype=np.float64)
        dep_sum = np.zeros(len(unique_bins), dtype=np.float64)
        counts = np.zeros(len(unique_bins), dtype=np.float64)
        np.add.at(indep_sum, inverse, indep_samples)
        np.add.at(dep_sum, inverse, dep_samples)
        np.add.at(counts, inverse, 1.0)
        indep = indep_sum / counts
        dep = dep_sum / counts
        order = np.argsort(indep)
        return indep[order], dep[order]

    @staticmethod
    def _bbox_limits(img_w: int, img_h: int) -> tuple[int, int, int, int, int]:
        min_w = 100
        min_h = 100
        max_w = 1000
        max_h = 1000
        min_area = 350
        return min_w, min_h, max_w, max_h, min_area

    @classmethod
    def _is_valid_detection_bbox(
        cls,
        bbox_w: int,
        bbox_h: int,
        img_w: int,
        img_h: int,
    ) -> bool:
        min_w, min_h, max_w, max_h, min_area = cls._bbox_limits(img_w, img_h)
        bbox_area = bbox_w * bbox_h
        return (
            bbox_w >= min_w
            and bbox_h >= min_h
            and bbox_w <= max_w
            and bbox_h <= max_h
            and bbox_area >= min_area
        )

    @classmethod
    def _fit_spline_axis(
        cls,
        indep_samples: np.ndarray,
        dep_samples: np.ndarray,
        residual_threshold: float,
        bin_size: float,
    ) -> tuple[UnivariateSpline, np.ndarray, np.ndarray, int]:
        indep, dep = cls._collapse_axis_samples(indep_samples, dep_samples, bin_size)
        if len(indep) < 2:
            raise ValueError("not enough unique samples for spline fit")

        # Lane dashes are locally smooth and mostly straight near the vehicle.
        # Capping the order at quadratic avoids cubic over-bending between sparse samples.
        spline_order = min(2, len(indep) - 1)
        smooth = 0.0 if len(indep) <= spline_order + 1 else len(indep) * (residual_threshold * 0.35) ** 2
        spline0 = UnivariateSpline(indep, dep, k=spline_order, s=smooth, ext=0)

        residuals = np.abs(dep_samples - spline0(indep_samples))
        inliers = residuals <= residual_threshold
        if np.count_nonzero(inliers) >= 2:
            indep, dep = cls._collapse_axis_samples(
                indep_samples[inliers], dep_samples[inliers], bin_size
            )
            if len(indep) < 2:
                raise ValueError("not enough inlier samples for spline fit")
            spline_order = min(2, len(indep) - 1)

        smooth = 0.0 if len(indep) <= spline_order + 1 else len(indep) * (residual_threshold * 0.25) ** 2
        spline = UnivariateSpline(indep, dep, k=spline_order, s=smooth, ext=0)
        return spline, indep, dep, spline_order

    @staticmethod
    def _fit_local_tangent(
        indep: np.ndarray,
        dep: np.ndarray,
        ref_value: float,
        min_window: float,
        anchor_on_low_end: bool = False,
    ) -> tuple[float, float]:
        if len(indep) < 2:
            raise ValueError("not enough samples for local tangent")

        indep = np.asarray(indep, dtype=np.float64)
        dep = np.asarray(dep, dtype=np.float64)
        span = float(max(indep.max() - indep.min(), 1e-6))
        window = max(min_window, 0.2 * span)

        if anchor_on_low_end:
            anchor = float(indep.min())
            mask = indep <= anchor + window
        else:
            anchor = float(indep.max())
            mask = indep >= anchor - window

        local_indep = indep[mask]
        local_dep = dep[mask]
        if len(local_indep) < 2:
            if anchor_on_low_end:
                local_indep = indep[:2]
                local_dep = dep[:2]
            else:
                local_indep = indep[-2:]
                local_dep = dep[-2:]

        if len(local_indep) == 2:
            slope = float((local_dep[1] - local_dep[0]) / (local_indep[1] - local_indep[0]))
            value = float(local_dep[0] + slope * (ref_value - local_indep[0]))
            return value, slope

        denom = max(float(local_indep.max() - local_indep.min()), 1e-6)
        progress = (local_indep - local_indep.min()) / denom
        if anchor_on_low_end:
            weights = 2.0 - progress
        else:
            weights = 1.0 + progress
        slope, intercept = np.polyfit(local_indep, local_dep, 1, w=weights)
        value = float(slope * ref_value + intercept)
        return value, float(slope)

    @staticmethod
    def _quadratic_coeffs_from_spline(
        spline: UnivariateSpline,
        ref_value: float,
    ) -> list[float]:
        x0 = float(spline(ref_value))
        dx = float(spline.derivative(1)(ref_value))
        try:
            ddx = float(spline.derivative(2)(ref_value))
        except ValueError:
            ddx = 0.0

        a = 0.5 * ddx
        b = dx - ddx * ref_value
        c = x0 - dx * ref_value + 0.5 * ddx * (ref_value ** 2)
        return [a, b, c]

    def _deproject_to_xz(
        self,
        image_pts: list[tuple[float, float]],
        depth_img: np.ndarray,
        cam_intr: dict[str, float],
        img_w: int,
        img_h: int,
    ) -> list[tuple[float, float, float, float]]:
        """Map RGB image points to camera-frame (X, Z) using the latest depth image."""
        dh, dw = depth_img.shape[:2]
        fx, cx = cam_intr["fx"], cam_intr["cx"]
        su = dw / float(img_w)
        sv = dh / float(img_h)
        samples: list[tuple[float, float, float, float]] = []

        for u_rgb, v_rgb in image_pts:
            ud = int(np.clip(round(u_rgb * su), 0, dw - 1))
            vd = int(np.clip(round(v_rgb * sv), 0, dh - 1))
            patch = depth_img[max(0, vd - 1):min(dh, vd + 2), max(0, ud - 1):min(dw, ud + 2)]
            valid = patch[np.isfinite(patch)]
            valid = valid[(valid >= 0.15) & (valid <= 20.0)]
            if valid.size == 0:
                continue
            z_m = float(np.median(valid))
            x_m = float((u_rgb - cx) * z_m / fx)
            samples.append((float(u_rgb), float(v_rgb), x_m, z_m))

        return samples

    def _fit_line_frenet(
        self,
        polygons: list[np.ndarray],
        image_h: int,
        image_w: int,
        depth_img: np.ndarray,
        cam_intr: dict[str, float],
        other_polys: list[np.ndarray] | None = None,
    ) -> dict[str, Any] | None:
        """other_polys: polygon của class seg PHỤ (vd curb) — chỉ được chiếu
        sang (d, s) để hiển thị trên panel Frenet 2D (sample_frenet_m_other),
        KHÔNG tham gia fit spline/d/heading. Không có line target thì không có
        spline tham chiếu -> class phụ cũng không vẽ được (trả None như cũ)."""
        if len(polygons) < 1:
            return None

        # Collect multi-point samples from every dash + centroids for viz
        all_xs: list[float] = []
        all_ys: list[float] = []
        # raw samples per dash (image coords) for visualization
        dash_samples: list[list[tuple[float, float]]] = []
        centroids: list[tuple[float, float]] = []
        for poly in polygons:
            pts = self._sample_dash_points(poly, self.cfg.seg_y_step)
            dash_samples.append(pts)
            for x, y in pts:
                all_xs.append(x)
                all_ys.append(y)
            centroids.append((float(np.mean(poly[:, 0])), float(np.mean(poly[:, 1]))))

        if len(all_xs) < 2:
            return None

        xs = np.array(all_xs, dtype=np.float64)
        ys = np.array(all_ys, dtype=np.float64)

        if ys.max() - ys.min() < 10:
            return None

        try:
            spline_2d, ys_fit, xs_fit, spline_order = self._fit_spline_axis(
                ys, xs, residual_threshold=15.0, bin_size=1.0
            )
        except ValueError:
            return None

        # Measure geometry on the lowest observed lane samples instead of extrapolating
        # the spline to the image bottom. This suppresses large d/heading jumps when the
        # spline curves between sparse dash observations.
        y_ref = float(min(image_h - 1, int(round(float(ys_fit.max())))))
        line_x, s_slope = self._fit_local_tangent(
            ys_fit, xs_fit, y_ref, min_window=24.0, anchor_on_low_end=False
        )

        # d at the local reference row — Method 2: perpendicular projection onto N̂
        # T̂ = [s, 1]/norm,  N̂ = [1, -s]/norm  (rotate T̂ right 90° in image coords)
        # δ = vehicle − P0 = [veh_x − line_x, 0]  (same row, evaluated at y_ref)
        # d = N̂ · δ = (veh_x − line_x) / norm  → positive = vehicle RIGHT of reference
        veh_x   = self.cfg.cam_cx if self.cfg.cam_cx > 0.0 else image_w / 2.0
        norm    = float(np.sqrt(1.0 + s_slope ** 2))
        d_pixels    = (line_x - veh_x) / norm   # positive = line RIGHT of vehicle (same convention as before)
        heading_px_deg = float(np.degrees(np.arctan(s_slope)))

        # Curve for overlay drawing: from vehicle row to farthest inlier sample
        y_far  = float(ys_fit.min())
        y_samp = np.linspace(y_ref, y_far, 80)
        curve_pts = np.column_stack([spline_2d(y_samp), y_samp]).astype(np.float32)
        coeffs = self._quadratic_coeffs_from_spline(spline_2d, y_ref)

        all_image_pts = [(sx, sy) for pts in dash_samples for sx, sy in pts]
        frenet: dict[str, Any] = {
            "d_pixels":    float(d_pixels),
            "heading_px_deg": heading_px_deg,
            "n_dashes":    len(polygons),
            "n_pts":       int(len(xs_fit)),
            "degree":      spline_order,
            "coeffs":      [float(c) for c in coeffs],
            "fit_mode":    "spline",
            "curve_pts":   curve_pts,
            "y_ref":       y_ref,
            "line_x":      line_x,
            "veh_x":       veh_x,
            "image_samples": all_image_pts,
        }

        metric_samples = self._deproject_to_xz(
            all_image_pts, depth_img, cam_intr, image_w, image_h
        )
        if len(metric_samples) < 2:
            return None

        xs_3d = np.array([p[2] for p in metric_samples], dtype=np.float64)
        zs_3d = np.array([p[3] for p in metric_samples], dtype=np.float64)
        if zs_3d.max() - zs_3d.min() <= 0.2:
            return None

        try:
            spline_3d, zs_fit, xs_fit_3d, spline_order_3d = self._fit_spline_axis(
                zs_3d, xs_3d, residual_threshold=0.20, bin_size=0.05
            )
        except ValueError:
            return None

        inliers_3d = np.abs(xs_3d - spline_3d(zs_3d)) <= 0.20
        if np.count_nonzero(inliers_3d) >= 2:
            metric_samples = [pt for pt, keep in zip(metric_samples, inliers_3d) if keep]

        d_meters, slope_m = self._fit_local_tangent(
            zs_fit, xs_fit_3d, 0.0, min_window=0.35, anchor_on_low_end=True
        )
        heading_deg = float(np.degrees(np.arctan(slope_m)))
        sample_frenet_m = [
            (float(x_m - spline_3d(z_m)), float(z_m))
            for _, _, x_m, z_m in metric_samples
        ]
        centroid_metric = self._deproject_to_xz(
            centroids, depth_img, cam_intr, image_w, image_h
        )
        dash_frenet_m = [
            (float(x_m - spline_3d(z_m)), float(z_m))
            for _, _, x_m, z_m in centroid_metric
        ]

        # Class phụ (viz-only): chiếu qua CÙNG spline_3d của line target để
        # nằm chung hệ (d, s) với sample_frenet_m — không lọc inlier (không
        # có model hình học riêng cho curb), chỉ hiển thị điểm thô.
        sample_frenet_m_other: list[tuple[float, float]] = []
        if other_polys:
            other_img_pts = [
                (sx, sy)
                for poly in other_polys
                for sx, sy in self._sample_dash_points(poly, self.cfg.seg_y_step)
            ]
            other_metric = self._deproject_to_xz(
                other_img_pts, depth_img, cam_intr, image_w, image_h
            )
            sample_frenet_m_other = [
                (float(x_m - spline_3d(z_m)), float(z_m))
                for _, _, x_m, z_m in other_metric
            ]

        # s_max chỉ dùng scale panel vẽ (không vào planner) — gộp cả điểm
        # class phụ để chúng không bị cắt khỏi panel khi ở xa hơn line.
        s_max_m = max(
            max((s_m for _, s_m in sample_frenet_m), default=0.5),
            max((s_m for _, s_m in sample_frenet_m_other), default=0.5),
        ) * 1.15

        frenet.update(
            {
                "d_meters": d_meters,
                "heading_deg": heading_deg,
                "spline_order_metric": spline_order_3d,
                "sample_frenet_m": sample_frenet_m,
                "sample_frenet_m_other": sample_frenet_m_other,
                "dash_frenet_m": dash_frenet_m,
                "s_max_m": float(max(s_max_m, 0.5)),
                "depth_points": len(metric_samples),
                "_metric_spline": spline_3d,
            }
        )
        return frenet

    def _run_yolo_detection(
        self,
        bgr: np.ndarray,
        seg_mask: np.ndarray,
        depth_img: np.ndarray,
        cam_intr: dict[str, float],
        frenet: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        result = self.yolo.predict(
            source=bgr,
            verbose=False,
            conf=self.cfg.yolo_conf,
            iou=self.cfg.yolo_iou,
            classes=self._YOLO_CLASSES,
            device=self.device_name,
        )[0]

        detections: list[dict[str, Any]] = []
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return detections
        img_h, img_w = bgr.shape[:2]

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            x1_i = max(0, int(round(x1)))
            y1_i = max(0, int(round(y1)))
            x2_i = min(seg_mask.shape[1], int(round(x2)))
            y2_i = min(seg_mask.shape[0], int(round(y2)))
            if x2_i <= x1_i or y2_i <= y1_i:
                continue

            bbox_width = x2_i - x1_i
            bbox_height = y2_i - y1_i
            if not self._is_valid_detection_bbox(
                bbox_width, bbox_height, img_w, img_h
            ):
                continue

            class_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())
            if confidence < self.cfg.detection_conf_gate:
                continue
            patch = seg_mask[y1_i:y2_i, x1_i:x2_i]
            bbox_area = int((x2_i - x1_i) * (y2_i - y1_i))
            seg_pixels = int(np.count_nonzero(patch))
            seg_ratio = float(seg_pixels / bbox_area) if bbox_area > 0 else 0.0

            detections.append(
                {
                    "class_id": class_id,
                    "label": self.class_names.get(class_id, str(class_id)),
                    "confidence": confidence,
                    "bbox": {
                        "x1": x1_i,
                        "y1": y1_i,
                        "x2": x2_i,
                        "y2": y2_i,
                    },
                    "segmentation": {
                        "enabled": seg_pixels > 0,
                        "mode": "yolov8seg",
                        "seg_pixels": seg_pixels,
                        "seg_ratio": seg_ratio,
                    },
                }
            )

        self._annotate_detection_frenet(
            detections, depth_img, cam_intr, bgr.shape[1], bgr.shape[0]
        )
        return detections

    def _estimate_detection_metric(
        self,
        bbox: dict[str, int],
        depth_img: np.ndarray,
        cam_intr: dict[str, float],
        img_w: int,
        img_h: int,
    ) -> tuple[float, float, float, float] | None:
        dh, dw = depth_img.shape[:2]
        fx, cx = cam_intr["fx"], cam_intr["cx"]
        su = dw / float(img_w)
        sv = dh / float(img_h)

        x1 = int(bbox["x1"])
        y1 = int(bbox["y1"])
        x2 = int(bbox["x2"])
        y2 = int(bbox["y2"])
        if x2 <= x1 or y2 <= y1:
            return None

        box_w = x2 - x1
        box_h = y2 - y1
        if not self._is_valid_detection_bbox(box_w, box_h, img_w, img_h):
            return None

        u_rgb = 0.5 * (x1 + x2)
        v_rgb = 0.5 * (y1 + y2)

        u0 = max(0, int(np.floor(x1 * su)))
        u1 = min(dw, int(np.ceil(x2 * su)))
        v0 = max(0, int(np.floor(y1 * sv)))
        v1 = min(dh, int(np.ceil(y2 * sv)))
        if u1 <= u0 or v1 <= v0:
            return None

        patch = depth_img[v0:v1, u0:u1]
        valid = patch[np.isfinite(patch)]
        valid = valid[(valid >= 0.15) & (valid <= 20.0)]
        if valid.size == 0:
            return None

        z_m = float(
            np.percentile(
                valid,
                float(np.clip(self.cfg.obstacle_depth_percentile, 0.0, 100.0)),
            )
        )
        x_m = float((u_rgb - cx) * z_m / fx)
        return float(u_rgb), float(v_rgb), x_m, z_m

    def _annotate_detection_frenet(
        self,
        detections: list[dict[str, Any]],
        depth_img: np.ndarray,
        cam_intr: dict[str, float],
        img_w: int,
        img_h: int,
    ) -> None:
        """Chiếu obstacle vào frame XE (x_m = lệch ngang so với trục quang
        tâm camera/xe, s_m = khoảng cách tiến) — KHÔNG cần line/spline, nên
        vẫn chạy được khi mất line. d_m giữ tương thích ngược (= x_m, frame
        xe) cho code cũ đọc trực tiếp; pipeline live nên dùng x_m/x_m_filtered
        + cộng pose xe (EKF hoặc đo trực tiếp) làm trung gian quy về panel
        frame — xem planner_motion/logic.py: plan_from_state().
        """
        for detection in detections:
            metric = self._estimate_detection_metric(
                detection["bbox"], depth_img, cam_intr, img_w, img_h
            )
            if metric is None:
                continue

            u_rgb, v_rgb, x_m, z_m = metric
            detection["frenet"] = {
                "available": True,
                "u": float(u_rgb),
                "v": float(v_rgb),
                "x_m": float(x_m),
                "z_m": float(z_m),
                "d_m": float(x_m),
                "s_m": float(z_m),
            }

        self._stabilize_detection_frenet(detections)

    def _stabilize_detection_frenet(
        self, detections: list[dict[str, Any]]
    ) -> None:
        current_frame = self.frame_count
        available_track_ids = [
            track_id
            for track_id, track in self._obstacle_tracks.items()
            if current_frame - int(track["last_seen"]) <= self.cfg.obstacle_track_ttl
        ]
        used_track_ids: set[int] = set()

        for detection in detections:
            frenet = detection.get("frenet")
            if not frenet or not frenet.get("available"):
                continue

            best_track_id: int | None = None
            best_cost = float("inf")
            for track_id in available_track_ids:
                if track_id in used_track_ids:
                    continue
                track = self._obstacle_tracks[track_id]
                if int(track["class_id"]) != int(detection["class_id"]):
                    continue

                du = abs(float(frenet["u"]) - float(track["u"]))
                ds = abs(float(frenet["s_m"]) - float(track["s_m"]))
                dd = abs(float(frenet["d_m"]) - float(track["d_m"]))
                if (
                    du > self.cfg.obstacle_match_px
                    or ds > self.cfg.obstacle_match_s
                    or dd > self.cfg.obstacle_match_d
                ):
                    continue

                norm_u = du / max(self.cfg.obstacle_match_px, 1e-6)
                norm_s = ds / max(self.cfg.obstacle_match_s, 1e-6)
                norm_d = dd / max(self.cfg.obstacle_match_d, 1e-6)
                cost = float(
                    np.sqrt(norm_u * norm_u + norm_s * norm_s + norm_d * norm_d)
                )
                if cost > self.cfg.obstacle_match_cost_max:
                    continue
                if cost < best_cost:
                    best_cost = cost
                    best_track_id = track_id

            if best_track_id is None:
                track_id = self._next_obstacle_track_id
                self._next_obstacle_track_id += 1
                track = {
                    "class_id": int(detection["class_id"]),
                    "u": float(frenet["u"]),
                    "v": float(frenet["v"]),
                    "x_m": float(frenet["x_m"]),
                    "z_m": float(frenet["z_m"]),
                    "d_m": float(frenet["d_m"]),
                    "s_m": float(frenet["s_m"]),
                    "last_seen": current_frame,
                }
                self._obstacle_tracks[track_id] = track
            else:
                track_id = best_track_id
                track = self._obstacle_tracks[track_id]
                a = self.cfg.obstacle_alpha
                track["u"] = self._ema(float(track["u"]), float(frenet["u"]), a)
                track["v"] = self._ema(float(track["v"]), float(frenet["v"]), a)
                track["x_m"] = self._ema(float(track["x_m"]), float(frenet["x_m"]), a)
                track["z_m"] = self._ema(float(track["z_m"]), float(frenet["z_m"]), a)
                track["d_m"] = self._ema(float(track["d_m"]), float(frenet["d_m"]), a)
                track["s_m"] = self._ema(float(track["s_m"]), float(frenet["s_m"]), a)
                track["last_seen"] = current_frame

            used_track_ids.add(track_id)
            frenet["track_id"] = int(track_id)
            frenet["u_filtered"] = float(track["u"])
            frenet["v_filtered"] = float(track["v"])
            frenet["x_m_filtered"] = float(track["x_m"])
            frenet["z_m_filtered"] = float(track["z_m"])
            frenet["d_m_filtered"] = float(track["d_m"])
            frenet["s_m_filtered"] = float(track["s_m"])

        self._obstacle_tracks = {
            track_id: track
            for track_id, track in self._obstacle_tracks.items()
            if current_frame - int(track["last_seen"]) <= self.cfg.obstacle_track_ttl
        }

    def _build_frenet_viz(self, frenet: dict[str, Any]) -> dict[str, Any]:
        """Project the internal frenet dict into a JSON-serializable payload
        carrying everything the visualization node needs to draw overlays."""
        return {
            "d_pixels": frenet["d_pixels"],
            "heading_px_deg": frenet["heading_px_deg"],
            "d_filtered": frenet.get("d_filtered"),
            "heading_px_filtered": frenet.get("heading_px_filtered"),
            "d_meters": frenet["d_meters"],
            "heading_deg": frenet["heading_deg"],
            "d_meters_filtered": frenet.get("d_meters_filtered"),
            "heading_filtered": frenet.get("heading_filtered"),
            "veh_x": frenet["veh_x"],
            "line_x": frenet["line_x"],
            "y_ref": frenet["y_ref"],
            "curve_pts": frenet["curve_pts"].tolist(),
            "image_samples": frenet.get("image_samples", []),
            "sample_frenet_m": frenet.get("sample_frenet_m", []),
            "sample_frenet_m_other": frenet.get("sample_frenet_m_other", []),
            "dash_frenet_m": frenet.get("dash_frenet_m", []),
            "s_max_m": frenet.get("s_max_m", 0.5),
            "lane_width_m": self.cfg.lane_width_m,
        }

    def _build_visual_frame(
        self, bgr: np.ndarray, seg_mask: np.ndarray, seg_color: np.ndarray
    ) -> np.ndarray:
        """Blend segmentation lên RGB ngay tại perception, để visualization chỉ
        cần vẽ thêm box/panel — không phải tự ghép 3 ảnh rời rạc nữa."""
        frame = bgr.copy()
        blend = cv2.addWeighted(
            bgr, 1.0 - self.cfg.segmentation_alpha, seg_color, self.cfg.segmentation_alpha, 0.0
        )
        frame[seg_mask > 0] = blend[seg_mask > 0]
        return frame

    def _update_frenet(self, frenet: dict[str, Any] | None) -> None:
        if frenet is None:
            return
        a = self.cfg.frenet_alpha

        # EMA chỉ áp dụng cho hiển thị debug ở pixel-space (bar dưới ảnh).
        self._d_ema = self._ema(self._d_ema, float(frenet["d_pixels"]), a)
        frenet["d_filtered"]       = self._d_ema
        self._hdg_px_ema = self._ema(
            self._hdg_px_ema, float(frenet["heading_px_deg"]), a
        )
        frenet["heading_px_filtered"] = self._hdg_px_ema

        # d_meters/heading_deg (dùng cho control) truyền thẳng RAW — EMA
        # thuần thời gian không bù được chuyển động xe giữa 2 frame (gây
        # lag/lệch lúc cua). control_node.ekf đã fusion đúng với /odom
        # (predict + correct), đây là nơi duy nhất nên smoothing/dead-reckon.
        self._hdg_raw = float(frenet["heading_deg"])
        self._d_m_raw = float(frenet["d_meters"])
        frenet["heading_filtered"] = self._hdg_raw
        frenet["d_meters_filtered"] = self._d_m_raw

    @staticmethod
    def _ema(previous: float | None, current: float, alpha: float) -> float:
        if previous is None:
            return current
        return alpha * current + (1.0 - alpha) * previous
