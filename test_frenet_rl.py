"""Xem trực quan SAC (RL) và Frenet cost-based trên các kịch bản cố định —
mỗi phương pháp 1 cửa sổ panel chạy song song, cuối cùng in bảng + lưu plot
quỹ đạo từng kịch bản để tự đánh giá.

Cả 2 chạy CÙNG vòng mô phỏng giống robot (tham số trong
config/rl_car_params.yaml): pure pursuit 40 Hz, đường lập lại MỖI tick,
sim_predict như env train.
- SAC: đúng logic PlannerLogic._plan_rl trên xe (compare_rl_frenet.RLPlanner):
  gate "chỉ đưa vật cản cho policy khi cần" + RLPolicy (khung gương, chốt
  phía, tối đa 3 vật) + kiểm tra cứng; đường RL bị loại -> tick đó dùng Frenet
  (đếm ở cột "Frenet thay").
- Frenet: FrenetOptimalPlanner.plan() nguyên vẹn.
Thất bại = hết đường khả thi (planner trả None) hoặc xe cách vật <= robot_radius.

    python3 test_frenet_rl.py [model] [--set single|multi|hard|all] [--video DIR]

--video DIR: quay video từng kịch bản (DIR/<kịch bản>_SAC.mp4, _Frenet.mp4 và
_compare.mp4 ghép song song SAC | Frenet), H.264 nếu có ffmpeg. Tốc độ phát
= VIDEO_SPEED lần thời gian thực. Chạy được không cần màn hình.

Nhóm kịch bản (--set, mặc định multi):
  single — 1 vật quét ngang làn + đường trống + lệch heading ±20°
  multi  — 2-3 vật dựng tay (zigzag, cổng, cùng phía, …)
  hard   — kịch bản sinh ngẫu nhiên (sample_obstacle_layout, luôn khả thi),
           nhiều vật, lấy từ đánh giá 200 ca (model parity 150k): khe giữa 2
           vật (trước đây RL hỏng, Frenet qua), chỉ Frenet hết đường, cả hai
           qua, cả hai hết đường
"""

from __future__ import annotations

import argparse
import math
import multiprocessing as mp
import os

import numpy as np

from control.ekf import FrenetEKF
from control.pure_pursuit import compute_cmd_vel

from compare_rl_frenet import MULTI_SCENARIOS, FrenetPlanner, RLPlanner
from train_frenet_rl import (
    ENV_CONFIG,
    PLANNER_CFG,
    PP_CFG,
    PLOTS_DIR,
    MODELS_DIR,
    INTERACTIVE_PLOTS,
    PANEL_LANE_WIDTH_M,
    plt,
    cv2,
    OverlayRenderer,
    FrenetStraightEnv,
    render_frenet_panel,
    sim_predict,
    PANEL_W,
    PANEL_H,
)

MODEL_PATH = os.path.join(MODELS_DIR, "sac_frenet_straight_parity_s1_150k")

# Tầm dọc panel [m] khi vẽ/quay video (path robot dài 2-3 m).
PANEL_VIEW_M = 5.0

# Kịch bản: (tên, [(s tuyệt đối, d), ...], d0, psi0). +d = phải, -d = trái.
S_OBS = ENV_CONFIG["course_length_m"] / 2.0
_SINGLE = [("no_obstacle", [], 0.0, 0.0)]
for _d in (-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5):
    _tag = "center" if _d == 0.0 else (f"left{abs(_d):.1f}" if _d < 0 else f"right{_d:.1f}")
    _SINGLE.append((f"obstacle_{_tag}", [(S_OBS, _d)], 0.0, 0.0))
_SINGLE += [
    ("heading_offset_left", [], 0.0, math.radians(20.0)),
    ("heading_offset_right", [], 0.0, math.radians(-20.0)),
]
_MULTI = [(name, obs, 0.0, 0.0) for name, obs in MULTI_SCENARIOS]


def _generated(seed: int, name: str):
    """Kịch bản từ bộ sinh của env (cùng seed với đánh giá 200 ca: 50000+ep)."""
    env = FrenetStraightEnv(config={"canonical": False})
    env.reset(seed=seed)
    return (name, list(env._obstacles), float(env.ekf.state.d), float(env.ekf.state.psi))


def _hard():
    groups = (("khe_giua", (157, 178)), ("Frenet_fail", (10, 14)),
              ("both_ok", (39, 103)), ("both_fail", (13, 187)))
    return [_generated(50_000 + ep, f"gen_ep{ep}_{tag}") for tag, eps in groups for ep in eps]


# Kịch bản tiêu biểu cho bảng tổng kết báo cáo (--set report).
_REPORT = ("no_obstacle", "heading_offset_left", "obstacle_left0.5", "obstacle_center",
           "zigzag trái→phải", "zigzag sát (4 m)", "giữa + chặn trái",
           "gen_ep10_Frenet_fail", "gen_ep39_both_ok", "gen_ep187_both_fail")


def scenario_set(name: str):
    if name == "report":
        by_name = {sc[0]: sc for sc in _SINGLE + _MULTI + _hard()}
        return [by_name[n] for n in _REPORT]
    if name == "single":
        return _SINGLE
    if name == "multi":
        return _MULTI
    if name == "hard":
        return _hard()
    return _SINGLE + _MULTI + _hard()


# Tần suất THỰC SỰ VẼ LÊN MÀN HÌNH lại thấp hơn nhiều — draw_frenet_panel +
# Tk draw_idle/flush_events có overhead đáng kể (đo thật: vẽ 1 panel ~4ms,
# nhưng round-trip qua Tk event loop mỗi lần gọi mới là phần chậm). 10Hz vẫn
# mượt mắt (mắt người khó phân biệt >10-12fps với chuyển động chậm/mượt như
# này). Vật lý vẫn tích phân đúng 50Hz phía trên, chỉ BỎ BỚT khung hình vẽ.
RENDER_HZ = 10.0

# Tốc độ phát lại so với thời gian thực. Đo thật: pure pursuit clamp
# linear_x ở PurePursuitConfig.max_linear_speed=1.0 m/s (KHÔNG phải
# target_speed=2.0), nên 1 episode dài 60m tốn ~60s mô phỏng — 8 lượt (4
# kịch bản x SAC/cost-based) ở đúng 1x có thể mất 10-15+ phút. 4x vẫn mượt
# hơn hẳn bản cũ (vốn chỉ 1 khung/tick, nhảy cách quãng) mà xem hết trong
# vài phút. Đổi về 1.0 nếu muốn đúng thời gian thực.
PLAYBACK_SPEED = 4.0

# Video (--video): 1 khung mỗi 1/RENDER_HZ s mô phỏng, phát ở VIDEO_SPEED lần
# thời gian thực.
VIDEO_SPEED = 2.0


def _animate_substeps(live: "LivePanel", info: dict, title: str) -> None:
    """Phát lại chuyển động TRONG 1 step từ info["trace"] — pose (s, d, psi)
    thật sau mỗi tick pure pursuit (ENV_CONFIG["pp_rate_hz"]), do env/vòng
    cost-based ghi lại. Path/obstacle hiển thị giữ như đầu step. Chỉ VẼ mỗi
    render_every_n tick (RENDER_HZ) cho đỡ tốn overhead Tk."""
    trace = info.get("trace") or [(info["s_ego_before"], info["d_before"], info["psi_before"])]
    tick_dt = ENV_CONFIG["dt"] / len(trace)
    render_every_n = max(1, round(1.0 / (RENDER_HZ * tick_dt)))
    render_pause = render_every_n * tick_dt / PLAYBACK_SPEED
    for i, (s_now, d_now, psi_now) in enumerate(trace):
        if i != len(trace) - 1 and (i + 1) % render_every_n != 0:
            continue
        sub_info = dict(info)
        sub_info["d_before"] = d_now
        sub_info["psi_before"] = psi_now
        sub_info["s_ego_before"] = s_now
        live.update(sub_info, title=title, pause=render_pause)


class LivePanel:
    """1 cửa sổ panel sống CHO 1 PHƯƠNG PHÁP (SAC hoặc cost-based) — chạy
    liên tục xuyên suốt từ kịch bản đầu tới cuối (không bị tạo lại/ghi đè
    bởi phương pháp kia). main() tạo 2 instance riêng (SAC, cost-based) nên
    cả 2 cửa sổ mở song song, dễ so sánh trực tiếp. Không làm gì nếu không
    có display (INTERACTIVE_PLOTS=False)."""

    def __init__(self, name: str):
        self.name = name
        self.renderer = OverlayRenderer(lane_width_m=PANEL_LANE_WIDTH_M)
        # Env "phụ" chỉ để tái dùng _build_path khi vẽ quạt candidate minh
        # hoạ (giống VizCallback) — không rollout.
        self.path_builder_env = FrenetStraightEnv()
        if INTERACTIVE_PLOTS:
            self.fig, self.ax = plt.subplots(figsize=(3.8, 4.6))
            try:
                self.fig.canvas.manager.set_window_title(name)
            except Exception:
                pass  # backend không hỗ trợ đặt tên cửa sổ — bỏ qua, không quan trọng
            self.fig.show()

    video = None  # cv2.VideoWriter khi đang quay (start_video/stop_video)

    def start_video(self, path: str) -> None:
        self.video = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"),
                                     RENDER_HZ * VIDEO_SPEED, (PANEL_W, PANEL_H))

    def stop_video(self) -> None:
        if self.video is not None:
            self.video.release()
            self.video = None

    def update(self, info: dict, title: str, pause: float = 1.0 / RENDER_HZ) -> None:
        canvas = render_frenet_panel(info, self.renderer, self.path_builder_env, title=title,
                                     view_m=PANEL_VIEW_M)
        if canvas is None:
            return
        if self.video is not None:
            self.video.write(canvas)
        cv2.imwrite(os.path.join(PLOTS_DIR, f"test_live_{self.name}.png"), canvas)
        if not INTERACTIVE_PLOTS:
            return
        self.ax.clear()
        self.ax.imshow(canvas[:, :, ::-1])  # BGR (cv2) -> RGB (matplotlib)
        self.ax.axis("off")
        self.fig.tight_layout()
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        plt.pause(pause)


def run_scenario(planner, obstacles_abs, d0, psi0, live: "LivePanel", title: str):
    """Chạy 1 kịch bản với vòng 40 Hz giống robot. planner.plan(d, psi, v,
    obstacles_rel) trả FrenetPath hoặc None. Mỗi "step" hiển thị = dt (0.5 s)
    gồm n_sub tick; animation phát lại pose thật từng tick."""
    cfg = ENV_CONFIG
    planner.reset()
    ekf = FrenetEKF()
    ekf.x = np.array([d0, psi0])
    s_ego = 0.0
    v = PLANNER_CFG.target_speed
    n_sub = max(1, round(cfg["dt"] * cfg["pp_rate_hz"]))
    sub_dt = cfg["dt"] / n_sub
    traj_s, traj_d = [s_ego], [float(ekf.state.d)]
    min_dist, stuck, step_idx = math.inf, False, 0

    def dist_now():
        return min((math.hypot(so - s_ego, do - float(ekf.state.d)) for so, do in obstacles_abs),
                   default=math.inf)

    while s_ego < cfg["course_length_m"] and not stuck:
        d_first, psi_first, s_first = float(ekf.state.d), float(ekf.state.psi), s_ego
        first, trace = None, []
        for _k in range(n_sub):
            d, psi = float(ekf.state.d), float(ekf.state.psi)
            visible = [(so - s_ego, do) for so, do in obstacles_abs
                       if 0.0 <= so - s_ego <= cfg["vision_range_m"]]
            fp = planner.plan(d, psi, v, visible)
            if fp is None:
                stuck = True
                break
            linear_x, angular_z, target_s, target_d = compute_cmd_vel(fp.s, fp.d, 0.0, d, psi, v, PP_CFG)
            if first is None:
                first = (fp, target_s, target_d, linear_x, angular_z)
            sim_predict(ekf, linear_x, angular_z, sub_dt)
            s_ego += linear_x * math.cos(psi) * sub_dt
            trace.append((s_ego, float(ekf.state.d), float(ekf.state.psi)))
            min_dist = min(min_dist, dist_now())
        if first is None:
            break
        fp, target_s, target_d, linear_x, angular_z = first
        step_idx += 1
        traj_s.append(s_ego)
        traj_d.append(float(ekf.state.d))
        fallback = getattr(planner, "fallback_ticks", 0)
        info = {
            "path_d": fp.d.tolist(),
            "path_s": fp.s.tolist(),
            "d_before": d_first,
            "psi_before": psi_first,
            "s_ego_before": s_first,
            # Ti gần đúng từ mẫu thời gian cuối path — chỉ để vẽ quạt minh hoạ.
            "Ti": float(fp.t[-1] + PLANNER_CFG.dt) if len(fp.t) else PLANNER_CFG.min_t,
            "target_s": target_s,
            "target_d": target_d,
            "obstacles": obstacles_abs,
            "linear_x": linear_x,
            "angular_z": angular_z,
            "trace": trace,
        }
        extra = f" Frenet thay={fallback}" if hasattr(planner, "fallback_ticks") else ""
        _animate_substeps(live, info, title=f"{title} step={step_idx}{extra}")
        if abs(float(ekf.state.d)) > cfg["d_max_offset_m"]:
            break
    failed = stuck or min_dist <= PLANNER_CFG.robot_radius
    return dict(traj_s=traj_s, traj_d=traj_d, failed=failed, stuck=stuck, min_dist=min_dist,
                fallback=getattr(planner, "fallback_ticks", 0))


class _CountingRLPlanner(RLPlanner):
    """RLPlanner + đếm số tick phải dùng Frenet (đường RL bị loại)."""

    def reset(self):
        super().reset()
        self.fallback_ticks = 0
        if not hasattr(self, "_orig_plan"):
            self._orig_plan = self.planner.plan

            def counted(*a, **k):
                self.fallback_ticks += 1
                return self._orig_plan(*a, **k)
            self.planner.plan = counted


def _run_method_process(method: str, model_path: str, set_name: str, result_queue,
                        video_dir: str | None = None, horizon_s: float | None = None) -> None:
    """Chạy trong PROCESS CON RIÊNG (spawn) — cửa sổ live của riêng process
    này, chạy tuần tự qua toàn bộ kịch bản cho ĐÚNG 1 phương pháp, gửi kết quả
    về process cha qua Queue. 2 process chạy thật song song."""
    live = LivePanel(name=method)
    planner = (_CountingRLPlanner(model_path, planner_cfg=PLANNER_CFG, horizon_s=horizon_s)
               if method == "SAC" else FrenetPlanner(planner_cfg=PLANNER_CFG, horizon_s=horizon_s))
    results = []
    for name, obstacles, d0, psi0 in scenario_set(set_name):
        if video_dir:
            live.start_video(os.path.join(video_dir, f"{name}_{method}.raw.mp4"))
        results.append((name, run_scenario(planner, obstacles, d0, psi0, live, title=f"{name} ({method})")))
        live.stop_video()
    result_queue.put(results)
    if INTERACTIVE_PLOTS:
        # Gửi kết quả về cha XONG mới block chờ đóng cửa sổ.
        plt.show(block=True)


def _h264(src: str, dst: str) -> None:
    """mp4v (OpenCV) -> H.264 (mở được trong PowerPoint/Word/trình duyệt);
    không có ffmpeg thì giữ nguyên file mp4v."""
    import shutil
    import subprocess
    if shutil.which("ffmpeg") and subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", src, "-c:v", "libx264",
             "-pix_fmt", "yuv420p", dst]).returncode == 0:
        os.remove(src)
    else:
        os.replace(src, dst)


def _make_videos(video_dir: str, name: str, sac: dict, cost: dict) -> None:
    """Chuyển 2 video từng phương pháp sang H.264 và ghép song song
    SAC | Frenet (video ngắn hơn giữ khung cuối), thêm dải tiêu đề kết quả."""
    raws = {m: os.path.join(video_dir, f"{name}_{m}.raw.mp4") for m in ("SAC", "Frenet")}
    frames = {}
    for m, path in raws.items():
        cap, fs = cv2.VideoCapture(path), []
        while True:
            ok, f = cap.read()
            if not ok:
                break
            fs.append(f)
        cap.release()
        frames[m] = fs or [np.full((PANEL_H, PANEL_W, 3), 255, np.uint8)]
    n = max(len(fs) for fs in frames.values())
    header_h = 34
    raw_cmp = os.path.join(video_dir, f"{name}_compare.raw.mp4")
    out = cv2.VideoWriter(raw_cmp, cv2.VideoWriter_fourcc(*"mp4v"), RENDER_HZ * VIDEO_SPEED,
                          (2 * PANEL_W, PANEL_H + header_h))

    def verdict(r):
        if not r["failed"]:
            return f"QUA (cach vat {r['min_dist']:.2f} m)" if math.isfinite(r["min_dist"]) else "QUA"
        return "HET DUONG" if r["stuck"] else "VA CHAM"

    for i in range(n):
        row = np.hstack([frames["SAC"][min(i, len(frames["SAC"]) - 1)],
                         frames["Frenet"][min(i, len(frames["Frenet"]) - 1)]])
        head = np.full((header_h, 2 * PANEL_W, 3), 255, np.uint8)
        for x0, tag, r in ((0, "SAC (RL)", sac), (PANEL_W, "Frenet", cost)):
            cv2.putText(head, f"{tag}: {verdict(r)}", (x0 + 8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 120, 0) if not r["failed"] else (0, 0, 200), 1, cv2.LINE_AA)
        out.write(np.vstack([head, row]))
    out.release()
    for m, path in raws.items():
        _h264(path, os.path.join(video_dir, f"{name}_{m}.mp4"))
    _h264(raw_cmp, os.path.join(video_dir, f"{name}_compare.mp4"))


def _plot_scenario(name: str, obstacles, sac, cost) -> None:
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=1, label="reference line")
    if obstacles:
        ax.scatter([o[0] for o in obstacles], [o[1] for o in obstacles], c="red", marker="x", s=90,
                   label="obstacle")
        for so, do in obstacles:
            ax.add_patch(plt.Circle((so, do), PLANNER_CFG.robot_radius, color="red", alpha=0.12))
    ax.plot(cost["traj_s"], cost["traj_d"], c="tab:orange", label="Frenet cost-based")
    ax.plot(sac["traj_s"], sac["traj_d"], c="tab:blue", label="SAC (+ Frenet khi đường RL bị loại)")
    ax.set_xlabel("s [m]")
    ax.set_ylabel("d [m]  (+ phải, − trái)")
    ax.set_ylim(-2.6, 2.6)
    ax.set_title(f"Scenario: {name}")
    ax.legend(loc="upper right", fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, f"test_{name}.png"))
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="?", default=MODEL_PATH)
    ap.add_argument("--set", default="multi", choices=("single", "multi", "hard", "all", "report"))
    ap.add_argument("--video", default=None, metavar="DIR", help="quay video từng kịch bản vào DIR")
    ap.add_argument("--horizon", type=float, default=None, metavar="T",
                    help="ép cả SAC và Frenet dùng cùng Ti=T [s] (cùng độ dài path)")
    args = ap.parse_args()
    os.makedirs(PLOTS_DIR, exist_ok=True)
    if args.video:
        os.makedirs(args.video, exist_ok=True)
    scenarios = scenario_set(args.set)
    print(f"[RUN] model: {args.model} | nhóm kịch bản: {args.set} ({len(scenarios)})"
          f" | horizon: {args.horizon if args.horizon else 'theo planner'}", flush=True)

    # 2 process OS riêng (spawn — an toàn với Tk/matplotlib), mỗi process 1
    # phương pháp, cửa sổ live riêng, chạy song song.
    ctx = mp.get_context("spawn")
    q_sac: "mp.Queue" = ctx.Queue()
    q_cost: "mp.Queue" = ctx.Queue()
    p_sac = ctx.Process(target=_run_method_process, args=("SAC", args.model, args.set, q_sac, args.video, args.horizon))
    p_cost = ctx.Process(target=_run_method_process, args=("Frenet", args.model, args.set, q_cost, args.video, args.horizon))
    p_sac.start()
    p_cost.start()
    sac_by_name = dict(q_sac.get())
    cost_by_name = dict(q_cost.get())

    header = (f"{'scenario':<26}{'planner':<8}{'thất bại':<10}{'hết đường':<11}"
              f"{'min_dist':<10}{'mean|d|':<9}{'Frenet thay':<12}")
    print(header)
    print("-" * len(header))
    totals = {"SAC": 0, "Frenet": 0}
    for name, obstacles, _d0, _psi0 in scenarios:
        sac, cost = sac_by_name[name], cost_by_name[name]
        _plot_scenario(name, obstacles, sac, cost)
        if args.video:
            _make_videos(args.video, name, sac, cost)
        for tag, r in (("SAC", sac), ("Frenet", cost)):
            totals[tag] += r["failed"]
            md = f"{r['min_dist']:.2f}" if math.isfinite(r["min_dist"]) else "—"
            fb = str(r["fallback"]) if tag == "SAC" else "—"
            print(f"{name:<26}{tag:<8}{str(r['failed']):<10}{str(r['stuck']):<11}{md:<10}"
                  f"{float(np.mean(np.abs(r['traj_d']))):<9.3f}{fb:<12}")
    for tag, n in totals.items():
        print(f"\n{tag}: thất bại {n}/{len(scenarios)} kịch bản")
    print(f"\nPlots lưu trong {PLOTS_DIR}/test_<scenario>.png")
    if args.video:
        print(f"Video lưu trong {args.video}/<kịch bản>_{{SAC,Frenet,compare}}.mp4")
    if INTERACTIVE_PLOTS:
        print("Xem xong — đóng cả 2 cửa sổ panel để 2 process kết thúc.")
    p_sac.join()
    p_cost.join()


if __name__ == "__main__":
    main()
