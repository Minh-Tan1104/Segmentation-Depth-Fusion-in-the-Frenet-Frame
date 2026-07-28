#!/usr/bin/env python3
"""Sandbox mô phỏng OFFLINE cho Frenet Optimal Planner.

Import THẲNG code thật (planner_motion/frenet_planner.py — không phải bản chép
lại), không import rclpy, không mở serial/camera/socket nào — chạy hoàn toàn
độc lập, không đụng tới process ROS (control_node/planner_motion_node) đang
chạy thật. Dùng để tinh chỉnh plan_speed/plan_center_weight/horizon/obstacle_*
bằng mắt trước khi sửa config/rl_car_params.yaml.

Chạy:
    python3 scripts/planner_sandbox.py

Giá trị khởi tạo lấy từ chính config/rl_car_params.yaml (khối /** dùng chung)
— kéo slider chỉnh thử, bấm "In YAML" để in ra đoạn yaml khớp giá trị đang xem,
copy dán lại vào rl_car_params.yaml khi ưng ý.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import yaml
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, CheckButtons, Slider

from planner_motion.frenet_planner import FrenetOptimalPlanner, FrenetPlannerConfig

YAML_PATH = REPO_ROOT / "config" / "rl_car_params.yaml"


def load_yaml_defaults() -> dict:
    """Đọc giá trị plan_* thật đang cấu hình (khối /** dùng chung cho
    control_node + planner_motion_node) — sandbox khởi động đúng ngay trạng
    thái hiện tại, không phải số bịa."""
    try:
        with open(YAML_PATH) as f:
            doc = yaml.safe_load(f)
        return dict(doc.get("/**", {}).get("ros__parameters", {}))
    except Exception as exc:
        print(f"[canh bao] khong doc duoc {YAML_PATH}: {exc} -> dung default cung")
        return {}


DEFAULTS = load_yaml_defaults()


def g(key: str, fallback: float) -> float:
    return float(DEFAULTS.get(key, fallback))


# Trạng thái sandbox — trùng tên với plan_* trong yaml để copy-paste trực tiếp.
# c_d/heading_deg/obstacle_* KHÔNG có trong yaml (đó là "tình huống" đang test,
# không phải tham số hệ thống).
P: dict[str, float | bool] = dict(
    c_d=0.8,                 # lệch ngang xuất phát [m], +d = phải (panel)
    heading_deg=20.0,        # heading xuất phát so với reference [deg]
    plan_speed=g("plan_speed", 2.0),
    plan_center_weight=g("plan_center_weight", 1.0),      # k_d
    plan_min_horizon_s=g("plan_min_horizon_s", 3.5),
    plan_max_horizon_s=g("plan_max_horizon_s", 4.0),
    plan_obstacle_weight=g("plan_obstacle_weight", 10.0),  # k_obs
    plan_clearance=g("plan_clearance", 1.2),
    plan_robot_radius=g("plan_robot_radius", 0.6),
    plan_road_width=g("plan_road_width", 2.5),
    plan_max_curvature=g("plan_max_curvature", 10.0),
    plan_center_offset=g("plan_center_offset", 0.0),
    plan_d_road_w=g("plan_d_road_w", 0.4),   # bước nhảy lấy mẫu di — nhỏ hơn = di mịn hơn, chậm hơn
    obstacle_s=3.0,           # vị trí vật cản dọc [m]
    obstacle_d=-0.3,          # vị trí vật cản ngang [m]
    obstacle_on=True,
)


def build_config(p: dict) -> FrenetPlannerConfig:
    return FrenetPlannerConfig(
        target_speed=p["plan_speed"],
        max_speed=p["plan_speed"] * 1.5,
        robot_radius=p["plan_robot_radius"],
        max_road_width=p["plan_road_width"],
        max_curvature=p["plan_max_curvature"],
        clearance=p["plan_clearance"],
        k_obs=p["plan_obstacle_weight"],
        center_offset=p["plan_center_offset"],
        k_d=p["plan_center_weight"],
        min_t=p["plan_min_horizon_s"],
        max_t=p["plan_max_horizon_s"],
        d_road_w=p["plan_d_road_w"],
    )


def run_planner(p: dict):
    """Y hệt cách control_node/planner_motion_node gọi thật (xem
    planner_motion/logic.py: c_speed = target_speed, KHÔNG lấy từ EKF.state.v
    — hardcode plan_speed, đúng hành vi live hiện tại kể cả khi có vẻ lạ)."""
    cfg = build_config(p)
    planner = FrenetOptimalPlanner(cfg)
    c_speed = p["plan_speed"]
    c_d_d = c_speed * math.sin(math.radians(p["heading_deg"]))
    obstacles = [(p["obstacle_s"], p["obstacle_d"])] if p["obstacle_on"] else None
    best, candidates = planner.plan(0.0, c_speed, p["c_d"], c_d_d, 0.0, obstacles)
    return cfg, best, candidates, obstacles


# ------------------------------------------------------------------ #
# Layout
# ------------------------------------------------------------------ #
fig = plt.figure(figsize=(15, 9))
fig.canvas.manager.set_window_title("RL_CAR — Frenet planner sandbox (offline, khong dung ROS)")

# Cột trái (x 0.03-0.31): toàn bộ điều khiển. Cột phải (x 0.37-0.97): đồ thị.
# 2 cột KHÔNG chồng x lên nhau — tránh slider đè lên plot.
ax_path = fig.add_axes([0.37, 0.42, 0.60, 0.53])    # d(x)-s(y), giống panel thật
ax_kappa = fig.add_axes([0.37, 0.07, 0.60, 0.27])   # curvature profile theo s
ax_info = fig.add_axes([0.03, 0.03, 0.28, 0.24])
ax_info.axis("off")

SLIDER_SPECS = [
    ("c_d", "c_d xuat phat [m]", -2.0, 2.0),
    ("heading_deg", "heading xuat phat [deg]", -45.0, 45.0),
    ("plan_speed", "plan_speed [m/s]", 0.5, 5.0),
    ("plan_center_weight", "plan_center_weight (k_d)", 0.1, 10.0),
    ("plan_min_horizon_s", "plan_min_horizon_s [s]", 1.0, 8.0),
    ("plan_max_horizon_s", "plan_max_horizon_s [s]", 1.5, 10.0),
    ("plan_obstacle_weight", "plan_obstacle_weight (k_obs)", 0.0, 30.0),
    ("plan_clearance", "plan_clearance [m]", 0.2, 3.0),
    ("plan_robot_radius", "plan_robot_radius [m]", 0.1, 1.5),
    ("plan_d_road_w", "plan_d_road_w [m]", 0.05, 1.0),
    ("obstacle_s", "obstacle_s [m]", 0.5, 8.0),
    ("obstacle_d", "obstacle_d [m]", -2.0, 2.0),
]

sliders: dict[str, Slider] = {}
slider_top = 0.94
slider_h = 0.043
for i, (key, label, lo, hi) in enumerate(SLIDER_SPECS):
    ax_s = fig.add_axes([0.09, slider_top - i * slider_h, 0.24, slider_h * 0.5])
    s = Slider(ax_s, label, lo, hi, valinit=float(P[key]), valstep=None)
    s.label.set_fontsize(7.5)
    s.label.set_position((0, 1.4))
    s.label.set_horizontalalignment("left")
    s.valtext.set_fontsize(7.5)
    sliders[key] = s
sliders_bottom = slider_top - len(SLIDER_SPECS) * slider_h  # đáy khối slider, đặt checkbox/nút dưới đây

ax_check = fig.add_axes([0.03, sliders_bottom - 0.06, 0.28, 0.05])
check = CheckButtons(ax_check, ["vat can bat"], [P["obstacle_on"]])

ax_reset = fig.add_axes([0.03, sliders_bottom - 0.13, 0.13, 0.05])
btn_reset = Button(ax_reset, "Reset ve yaml")

ax_print = fig.add_axes([0.18, sliders_bottom - 0.13, 0.13, 0.05])
btn_print = Button(ax_print, "In YAML")


def redraw(_event=None) -> None:
    for key in sliders:
        P[key] = sliders[key].val
    P["obstacle_on"] = check.get_status()[0]

    cfg, best, candidates, obstacles = run_planner(P)

    ax_path.cla()
    ax_kappa.cla()
    ax_info.cla()
    ax_info.axis("off")

    d_disp = P["plan_road_width"] + 0.5
    ax_path.set_xlim(-d_disp, d_disp)
    ax_path.set_ylim(-0.5, max(6.0, P["plan_speed"] * P["plan_max_horizon_s"] * 1.2))
    ax_path.axvline(0.0, color="tab:green", lw=1, alpha=0.6, label="reference (d=0)")
    ax_path.axvline(P["plan_center_offset"], color="tab:green", lw=1, ls="--", alpha=0.4)
    for sign in (-1, 1):
        ax_path.axvline(sign * P["plan_road_width"], color="0.6", lw=0.8, ls=":")
    ax_path.axhline(0.0, color="0.7", lw=0.6)

    for fp in candidates:
        if len(fp.d) < 2:
            continue
        ax_path.plot(fp.d, fp.s, color="0.75", lw=0.6, zorder=1)

    if obstacles:
        for os_, od_ in obstacles:
            circ_hard = plt.Circle((od_, os_), P["plan_robot_radius"], color="red", alpha=0.18, zorder=2)
            circ_soft = plt.Circle((od_, os_), P["plan_clearance"], color="orange", fill=False, lw=1, ls="--", zorder=2)
            ax_path.add_patch(circ_hard)
            ax_path.add_patch(circ_soft)
            ax_path.plot(od_, os_, "x", color="red", zorder=3)

    # xe: mui chi huong theo heading xuat phat
    hdg = math.radians(P["heading_deg"])
    ax_path.plot(P["c_d"], 0.0, "o", color="tab:blue", zorder=4, markersize=8)
    ax_path.annotate(
        "", xy=(P["c_d"] + 0.6 * math.sin(hdg), 0.6 * math.cos(hdg)),
        xytext=(P["c_d"], 0.0),
        arrowprops=dict(arrowstyle="->", color="tab:blue", lw=2), zorder=4,
    )

    if best is not None and len(best.d):
        ax_path.plot(best.d, best.s, color="tab:cyan", lw=2.5, zorder=5, label="path chon (best)")
        # best.c (curvature) ngắn hơn best.s đúng 1 phần tử — frenet_planner.py
        # tính c qua np.diff giữa N điểm liên tiếp -> N-1 đoạn. Cắt s theo cho khớp.
        s_arr, kappa = best.s[: len(best.c)], best.c
        ax_kappa.plot(s_arr, np.abs(kappa), color="tab:cyan", lw=1.8)
        ax_kappa.axhline(P["plan_max_curvature"], color="red", lw=1, ls="--", label="plan_max_curvature")
        k_early = float(np.max(np.abs(kappa[: max(1, len(kappa) // 4)])))
        k_max = float(np.max(np.abs(kappa))) if len(kappa) else 0.0
        status = "KHONG co path kha thi (kiem tra robot_radius/max_curvature)" if best is None else ""
        info = (
            f"di chon (d cuoi)   = {best.d[-1]:+.3f} m\n"
            f"s_max path         = {best.s[-1]:.2f} m\n"
            f"|kappa| doan dau*  = {k_early:.3f}  (* 1/4 dau path)\n"
            f"|kappa| max toan bo= {k_max:.3f}\n"
            f"gioi han cung      = {P['plan_max_curvature']:.2f}\n"
            f"con lai            = {P['plan_max_curvature']-k_max:+.2f}\n"
        )
    else:
        info = "!!! KHONG co quy dao kha thi !!!\n(vat can/robot_radius/max_curvature\ndang loai het ung vien)"
        k_max = float("nan")

    ax_kappa.set_xlabel("s [m]")
    ax_kappa.set_ylabel("|kappa| [1/m]")
    ax_kappa.legend(fontsize=7, loc="upper right")
    ax_kappa.grid(alpha=0.3)

    ax_path.set_xlabel("d [m]  (+ = phai)")
    ax_path.set_ylabel("s [m]")
    ax_path.legend(fontsize=8, loc="upper left")
    ax_path.grid(alpha=0.3)
    ax_path.set_title(
        f"c_speed dung sinh path = plan_speed = {P['plan_speed']:.2f} m/s"
        f"   (auto_speed lai that co the KHAC — xem README#luu-y)",
        fontsize=9,
    )

    ax_info.text(0.0, 1.0, "=== SO LIEU ===\n" + info, va="top", ha="left",
                 family="monospace", fontsize=9, transform=ax_info.transAxes)

    fig.canvas.draw_idle()


def on_reset(_event) -> None:
    fresh = load_yaml_defaults()
    for key, slider in sliders.items():
        if key in fresh:
            slider.set_val(float(fresh[key]))
    redraw()


def on_print(_event) -> None:
    print("\n# --- dan doan nay vao khoi /** trong config/rl_car_params.yaml ---")
    for key in (
        "plan_speed", "plan_center_weight", "plan_min_horizon_s",
        "plan_max_horizon_s", "plan_obstacle_weight", "plan_clearance",
        "plan_robot_radius", "plan_road_width", "plan_max_curvature",
        "plan_center_offset", "plan_d_road_w",
    ):
        print(f"    {key}: {P[key]}")
    print(f"# (tinh huong dang xem, KHONG phai param yaml): "
          f"c_d={P['c_d']}, heading_deg={P['heading_deg']}, "
          f"obstacle=({P['obstacle_s']},{P['obstacle_d']}) on={P['obstacle_on']}\n")


for s in sliders.values():
    s.on_changed(redraw)
check.on_clicked(redraw)
btn_reset.on_clicked(on_reset)
btn_print.on_clicked(on_print)

redraw()
plt.show()
