"""Test SAC agent (đã train bằng train_frenet_rl.py) trên 4 kịch bản cố định
và so sánh với cost-based Frenet planner gốc (planner_motion/frenet_planner.py,
gọi lại đúng FrenetOptimalPlanner.plan() — không viết lại cost).

Mô phỏng giống env train (train_frenet_rl.py): tham số planner/pure pursuit
của robot, mỗi step 0.5 s gồm các tick pure pursuit 40 Hz và đường được lập
lại mỗi tick. Animation phát lại pose thật của từng tick (info["trace"]).

Khi so sánh: dùng FrenetStraightEnv(discretize_d_for_eval=True) để round
d_target liên tục của SAC về lưới rời rạc (center_offset + k*d_road_w) mà
cost-based planner đang dùng — so sánh công bằng (xem train_frenet_rl.py:
_round_to_grid).

SAC chạy qua ĐÚNG đường của robot: RLPolicy (planner_motion/rl_policy.py —
đọc meta cạnh model, tự áp khung gương + chốt phía nếu model train theo
kiểu đó) + gate "chỉ đưa vật cản cho policy khi đi thẳng không an toàn"
(như PlannerLogic._plan_rl). Nên chạy được cả model mới lẫn model cũ.

    python3 test_frenet_rl.py [đường_dẫn_model]   # mặc định MODEL_PATH
"""

from __future__ import annotations

import math
import multiprocessing as mp
import os
import sys

import numpy as np

from control.ekf import FrenetEKF
from control.pure_pursuit import compute_cmd_vel
from planner_motion.frenet_planner import FrenetOptimalPlanner
from planner_motion.rl_policy import RLPolicy, decode_action

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
)

MODEL_PATH = os.path.join(MODELS_DIR, "sac_frenet_straight_robot_s1_50k")
# Gate như control_node (PlannerLogic._plan_rl): False = policy luôn thấy vật cản.
USE_GATE = True

# Kịch bản cố định: obstacle đặt giữa đoạn đường (d0/psi0=0), quét NGANG qua
# 7 vị trí d_obs cách nhau 0.5m (thay vì 3 vị trí cách 1.5m trước đây) để
# đánh giá dày hơn qua toàn bộ bề ngang làn — từ -1.5 tới +1.5m (+d = phải,
# nên trái là d âm). + 2 kịch bản KHÔNG obstacle nhưng heading lệch sẵn
# (psi0=±20°) để thấy rõ dao động hội tụ của Pure Pursuit (đã verify bằng
# script debug riêng: sau khi sửa dấu v_odom trong
# FrenetStraightEnv.step()/run_cost_based, heading dao động TẮT DẦN về 0 —
# không còn phân kỳ). (name, obstacle, d0, psi0).
S_OBS = ENV_CONFIG["course_length_m"] / 2.0
_OBSTACLE_D_VALUES = [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5]  # 0.5m/bước, giữa lane ±2.4m
SCENARIOS = [("no_obstacle", None, 0.0, 0.0)]
for _d in _OBSTACLE_D_VALUES:
    _tag = "center" if _d == 0.0 else (f"left{abs(_d):.1f}" if _d < 0 else f"right{_d:.1f}")
    SCENARIOS.append((f"obstacle_{_tag}", (S_OBS, _d), 0.0, 0.0))
SCENARIOS += [
    ("heading_offset_left", None, 0.0, math.radians(20.0)),
    ("heading_offset_right", None, 0.0, math.radians(-20.0)),
]

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

    def update(self, info: dict, title: str, pause: float = 1.0 / RENDER_HZ) -> None:
        canvas = render_frenet_panel(info, self.renderer, self.path_builder_env, title=title)
        if canvas is None:
            return
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


def _sac_action(policy: RLPolicy, env: FrenetStraightEnv) -> np.ndarray:
    """Action ở khung THẬT cho env, qua đúng đường của robot (xem docstring
    module). Obstacle đưa theo (ds tương đối, d) như PlannerLogic."""
    ds_obs, d_obs, ds_real = env._obstacle_obs()
    d, psi = float(env.ekf.state.d), float(env.ekf.state.psi)
    obstacles = [(ds_obs, d_obs)] if math.isfinite(ds_real) else []
    if obstacles and USE_GATE:
        free = policy.action(d, psi, [], latch=False)
        d_target, Ti = decode_action(free, policy.meta)
        v = env.planner_cfg.target_speed
        fp = env._build_path(d, v * math.sin(psi), d_target, Ti, v)
        planner = env._planner
        if planner._check_paths([fp], obstacles) and planner._obstacle_cost(fp, obstacles) == 0.0:
            return free
    return policy.action(d, psi, obstacles)


def run_sac(policy: RLPolicy, d0: float, psi0: float, obstacle, live: LivePanel, title_prefix: str):
    """Chạy 1 kịch bản SAC tới hết. Gọi trong process RIÊNG (xem
    _run_method_process/main) nên không cần interleave với cost-based nữa —
    mỗi process chỉ lo đúng 1 phương pháp, chạy thẳng tới hết."""
    # canonical=False: env nhận action khung THẬT — RLPolicy đã tự lật gương.
    env = FrenetStraightEnv(config={"canonical": False}, discretize_d_for_eval=True)
    env.reset(options={"d0": d0, "psi0": psi0, "obstacle": obstacle})
    policy.latch.reset()
    traj_s = [env.s_ego]
    traj_d = [float(env.ekf.state.d)]
    collided = False
    min_dist = math.inf
    step_idx = 0
    done = False
    while not done:
        _obs, _r, terminated, truncated, info = env.step(_sac_action(policy, env))
        step_idx += 1
        traj_s.append(env.s_ego)
        traj_d.append(float(env.ekf.state.d))
        min_dist = min(min_dist, info["closest_dist"])
        if info.get("collided"):
            collided = True
        _animate_substeps(live, info, title=f"{title_prefix} (SAC) step={step_idx}")
        done = terminated or truncated
    return np.array(traj_s), np.array(traj_d), collided, min_dist


def run_cost_based(d0: float, psi0: float, obstacle, live: LivePanel, title_prefix: str):
    """Baseline: gọi lại NGUYÊN VẸN FrenetOptimalPlanner.plan() (cost-based,
    enumerate di x Ti x tv) mỗi step, cùng nhịp mô phỏng/pure-pursuit/EKF
    như env RL để so sánh công bằng. Chạy trong process riêng — xem
    docstring run_sac()."""
    cfg = ENV_CONFIG
    planner = FrenetOptimalPlanner(PLANNER_CFG)
    ekf = FrenetEKF()
    ekf.x = np.array([d0, psi0])
    s_ego = 0.0
    traj_s = [s_ego]
    traj_d = [float(ekf.state.d)]
    collided = False
    min_dist = math.inf
    step_idx = 0

    n_sub = max(1, round(cfg["dt"] * cfg["pp_rate_hz"]))
    sub_dt = cfg["dt"] / n_sub
    c_speed = PLANNER_CFG.target_speed
    for _ in range(cfg["max_episode_steps"]):
        # 1 "step" hiển thị = dt; bên trong pure pursuit chạy pp_rate_hz và
        # Frenet lập đường lại MỖI tick (như robot, xem env RL).
        d_first, psi_first, s_first = float(ekf.state.d), float(ekf.state.psi), s_ego
        first = None
        trace = []
        for _k in range(n_sub):
            d = float(ekf.state.d)
            psi = float(ekf.state.psi)
            obstacles_relative = []
            if obstacle is not None:
                s_obs_abs, d_obs = obstacle
                obstacles_relative.append((s_obs_abs - s_ego, d_obs))
            best, _candidates = planner.plan(0.0, c_speed, d, c_speed * math.sin(psi), 0.0, obstacles_relative)
            if best is None:
                collided = True
                break
            min_dist = min(min_dist, FrenetStraightEnv._closest_obstacle_dist(best, obstacles_relative))
            linear_x, angular_z, target_s, target_d = compute_cmd_vel(
                best.s, best.d, 0.0, d, psi, c_speed, PP_CFG,
            )
            if first is None:
                first = (best, target_s, target_d, linear_x, angular_z)
            sim_predict(ekf, linear_x, angular_z, sub_dt)
            s_ego += linear_x * math.cos(psi) * sub_dt
            trace.append((s_ego, float(ekf.state.d), float(ekf.state.psi)))
        if first is None:
            break
        best, target_s, target_d, linear_x, angular_z = first
        step_idx += 1
        traj_s.append(s_ego)
        traj_d.append(float(ekf.state.d))

        # info tương thích render_frenet_panel — cost-based không có "Ti"
        # tường minh trên FrenetPath, suy ra gần đúng từ mẫu thời gian cuối
        # path (best.t[-1] + dt ≈ Ti) — chỉ để vẽ quạt minh hoạ.
        info = {
            "path_d": best.d.tolist(),
            "path_s": best.s.tolist(),
            "d_before": d_first,
            "psi_before": psi_first,
            "s_ego_before": s_first,
            "Ti": float(best.t[-1] + PLANNER_CFG.dt) if len(best.t) else PLANNER_CFG.min_t,
            "target_s": target_s,
            "target_d": target_d,
            "obstacle": obstacle,
            "linear_x": linear_x,
            "angular_z": angular_z,
            "trace": trace,
        }
        _animate_substeps(live, info, title=f"{title_prefix} (cost-based) step={step_idx}")

        if collided or abs(float(ekf.state.d)) > cfg["d_max_offset_m"]:
            break
        if s_ego >= cfg["course_length_m"]:
            break

    return np.array(traj_s), np.array(traj_d), collided, min_dist


def _run_method_process(method: str, model_path: str, result_queue) -> None:
    """Chạy trong PROCESS CON RIÊNG (spawn) — mở cửa sổ live CỦA RIÊNG
    process này (OverlayRenderer/FrenetStraightEnv không share được giữa
    process nên phải tạo lại), chạy tuần tự qua toàn bộ SCENARIOS cho ĐÚNG
    1 phương pháp, rồi gửi kết quả về process cha qua Queue. 2 process này
    chạy THỰC SỰ song song (2 tiến trình OS riêng, có thể trên 2 core khác
    nhau) — khác hẳn interleave trong 1 process (vẫn tuần tự ở mức CPU)."""
    live = LivePanel(name=method)
    model = RLPolicy(model_path) if method == "SAC" else None

    results = []
    for name, obstacle, d0, psi0 in SCENARIOS:
        if method == "SAC":
            traj_s, traj_d, collided, min_dist = run_sac(
                model, d0, psi0, obstacle, live, title_prefix=name
            )
        else:
            traj_s, traj_d, collided, min_dist = run_cost_based(
                d0, psi0, obstacle, live, title_prefix=name
            )
        results.append((name, traj_s.tolist(), traj_d.tolist(), collided, min_dist))
    result_queue.put(results)

    if INTERACTIVE_PLOTS:
        # Gửi kết quả về cha XONG mới block chờ đóng cửa sổ — cha không cần
        # đợi user đóng cửa sổ mới in được bảng/vẽ plot so sánh.
        plt.show(block=True)


def _plot_scenario(name: str, obstacle, sac_traj, cost_traj) -> None:
    sac_s, sac_d, _, _ = sac_traj
    cost_s, cost_d, _, _ = cost_traj
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=1, label="reference line")
    if obstacle is not None:
        s_o, d_o = obstacle
        ax.scatter([s_o], [d_o], c="red", marker="x", s=90, label="obstacle")
    ax.plot(cost_s, cost_d, c="tab:orange", label="cost-based planner")
    ax.plot(sac_s, sac_d, c="tab:blue", label="SAC (rounded to grid)")
    ax.set_xlabel("s [m]")
    ax.set_ylabel("d [m]")
    ax.set_title(f"Scenario: {name}")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, f"test_{name}.png"))
    plt.close(fig)


def main() -> None:
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # 2 PROCESS OS riêng (spawn — an toàn với Tk/matplotlib, tránh vấn đề
    # fork+GUI) — mỗi process chạy ĐÚNG 1 phương pháp qua toàn bộ SCENARIOS,
    # mở cửa sổ live của riêng nó. Chạy thật song song (không chỉ interleave
    # trong 1 process), có thể trên 2 core CPU khác nhau.
    ctx = mp.get_context("spawn")
    q_sac: "mp.Queue" = ctx.Queue()
    q_cost: "mp.Queue" = ctx.Queue()
    model_path = sys.argv[1] if len(sys.argv) > 1 else MODEL_PATH
    print(f"[RUN] model: {model_path}", flush=True)
    p_sac = ctx.Process(target=_run_method_process, args=("SAC", model_path, q_sac))
    p_cost = ctx.Process(target=_run_method_process, args=("cost-based", MODEL_PATH, q_cost))

    print("[RUN] 2 process song song: SAC + cost-based ...", flush=True)
    p_sac.start()
    p_cost.start()

    # get() chỉ chờ tới khi mỗi process GỬI xong kết quả (ngay sau vòng lặp
    # scenario) — KHÔNG cần đợi cửa sổ live của nó được đóng.
    sac_by_name = {r[0]: r for r in q_sac.get()}
    cost_by_name = {r[0]: r for r in q_cost.get()}

    rows = []
    for name, obstacle, d0, psi0 in SCENARIOS:
        _n, sac_s, sac_d, sac_collided, sac_min_dist = sac_by_name[name]
        _n, cost_s, cost_d, cost_collided, cost_min_dist = cost_by_name[name]
        sac_traj = (np.array(sac_s), np.array(sac_d), sac_collided, sac_min_dist)
        cost_traj = (np.array(cost_s), np.array(cost_d), cost_collided, cost_min_dist)
        _plot_scenario(name, obstacle, sac_traj, cost_traj)

        for tag, (traj_s, traj_d, collided, min_dist) in (
            ("SAC", sac_traj), ("cost-based", cost_traj),
        ):
            rows.append({
                "scenario": name,
                "planner": tag,
                "collided": collided,
                "mean_abs_d": float(np.mean(np.abs(traj_d))),
                "std_abs_d": float(np.std(np.abs(traj_d))),
                "min_dist_to_obstacle": min_dist if math.isfinite(min_dist) else float("nan"),
            })

    header = f"{'scenario':<22}{'planner':<12}{'collided':<10}{'mean|d|':<10}{'std|d|':<10}{'min_dist':<10}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['scenario']:<22}{r['planner']:<12}{str(r['collided']):<10}"
            f"{r['mean_abs_d']:<10.3f}{r['std_abs_d']:<10.3f}{r['min_dist_to_obstacle']:<10.3f}"
        )

    for tag in ("SAC", "cost-based"):
        n = sum(1 for r in rows if r["planner"] == tag)
        n_collided = sum(1 for r in rows if r["planner"] == tag and r["collided"])
        print(f"\n{tag}: tỉ lệ va chạm = {n_collided}/{n} kịch bản")

    print(f"\nPlots lưu trong {PLOTS_DIR}/test_<scenario>.png")
    if INTERACTIVE_PLOTS:
        print("Xem xong — đóng cả 2 cửa sổ panel để 2 process kết thúc.")
    p_sac.join()
    p_cost.join()


if __name__ == "__main__":
    main()
