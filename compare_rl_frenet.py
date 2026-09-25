"""So sánh RL (SAC) với Frenet cost-based trên 4 trường hợp: vật cản trái,
phải, giữa, không vật cản. 3 chỉ số:

- plan_ms:  thời gian chọn quỹ đạo mỗi tick (RL: RLPolicy + gate + build +
            check như PlannerLogic._plan_rl; Frenet: FrenetOptimalPlanner.plan).
- min_dist: khoảng cách nhỏ nhất từ XE (vị trí thật) tới vật cản [m].
- mean|d|:  độ lệch ngang trung bình so với tâm làn cả lượt chạy [m].

Hai phương pháp chạy CÙNG vòng mô phỏng (compute_cmd_vel + sim_predict của
train_frenet_rl.py) và cùng seed từng lượt: nhiễu đo
d_obs mỗi tick + lệch nhẹ điều kiện đầu. RL dùng action liên tục như trên xe.

    python3 compare_rl_frenet.py [model] [--runs N] [--rate HZ] [--multi]

--multi: kịch bản NHIỀU vật cản. Frenet chạy trước trên mọi kịch bản; chỉ
kịch bản Frenet qua được (không va, không hết đường) mới so với RL — tránh
kịch bản bất khả thi. Kịch bản Frenet không qua vẫn chạy RL, báo riêng.

Tham số planner/pure pursuit = tham số robot (config/rl_car_params.yaml, đọc
qua train_frenet_rl.robot_config). Pure pursuit ra lệnh ở --rate (mặc định
40 Hz) và planner lập đường lại MỖI tick (không giới hạn tần số). RL giải mã
action theo meta lúc train (Ti, tầm nhìn), như PlannerLogic trên xe.
"""

from __future__ import annotations

import argparse
import math
import time

import numpy as np

from control.ekf import FrenetEKF
from control.pure_pursuit import compute_cmd_vel
from planner_motion.frenet_planner import FrenetOptimalPlanner
from planner_motion.rl_policy import RLPolicy, lane_keeping_is_clear
from train_frenet_rl import ENV_CONFIG, PLANNER_CFG, PP_CFG, MODELS_DIR, sim_predict

import os

S_OBS = 30.0
# (tên, danh sách vật cản (s tuyệt đối, d)); +d = phải, -d = trái.
SCENARIOS = [
    ("vật cản trái", [(S_OBS, -0.5)]),
    ("vật cản phải", [(S_OBS, +0.5)]),
    ("vật cản giữa", [(S_OBS, 0.0)]),
    ("không vật cản", []),
]
MULTI_SCENARIOS = [
    ("zigzag trái→phải", [(25.0, -0.5), (35.0, +0.5)]),
    ("zigzag phải→trái", [(25.0, +0.5), (35.0, -0.5)]),
    ("cùng phía trái", [(25.0, -0.5), (35.0, -0.5)]),
    ("cổng giữa 2 vật", [(30.0, -1.2), (30.0, +1.2)]),
    ("giữa + chặn trái", [(30.0, 0.0), (30.0, -1.5)]),
    ("zigzag 3 vật", [(20.0, -0.5), (30.0, +0.5), (40.0, -0.5)]),
    ("zigzag sát (4 m)", [(28.0, -0.5), (32.0, +0.5)]),
    ("giữa rồi lệch phải", [(25.0, 0.0), (35.0, +1.0)]),
]
OBS_NOISE_M = 0.05   # nhiễu đo d_obs mỗi tick (perception)
INIT_D_M = 0.1       # |d0| <= 0.1 m
INIT_PSI_DEG = 3.0   # |psi0| <= 3°


class RLPlanner:
    def __init__(self, model_path: str, planner_cfg=PLANNER_CFG):
        self.policy = RLPolicy(model_path)
        self.planner = FrenetOptimalPlanner(planner_cfg)

    def reset(self):
        self.policy.latch.reset()

    def _build(self, d, c_d_d, v, action, obstacles):
        from planner_motion.rl_policy import decode_action
        d_target, Ti = decode_action(action, self.policy.meta)
        fp = self.planner.build_path(0.0, v, d, c_d_d, 0.0, d_target, Ti, v)
        ok = self.planner._check_paths(self.planner._calc_global_paths([fp]), obstacles)
        return ok[0] if ok else None

    def plan(self, d, psi, v, obstacles):
        """Đúng logic PlannerLogic._plan_rl (gate + fallback cost-based)."""
        c_d_d = v * math.sin(psi)
        if obstacles:
            fp = self._build(d, c_d_d, v, self.policy.action(d, psi, [], latch=False), obstacles)
            if fp is not None and lane_keeping_is_clear(
                    fp.s, fp.d, obstacles, self.planner.config.clearance, self.policy.meta.vision_range_m):
                return fp
        fp = self._build(d, c_d_d, v, self.policy.action(d, psi, obstacles), obstacles)
        if fp is None:
            fp, _ = self.planner.plan(0.0, v, d, c_d_d, 0.0, obstacles)
        return fp


class FrenetPlanner:
    def __init__(self, planner_cfg=PLANNER_CFG):
        self.planner = FrenetOptimalPlanner(planner_cfg)

    def reset(self):
        pass

    def plan(self, d, psi, v, obstacles):
        best, _ = self.planner.plan(0.0, v, d, v * math.sin(psi), 0.0, obstacles)
        return best


def run_once(method, obstacles_abs, seed, pp_cfg=PP_CFG, dt=1.0 / ENV_CONFIG["pp_rate_hz"], planner_cfg=PLANNER_CFG):
    """obstacles_abs: [(s tuyệt đối, d)]. Trả (va_chạm_hoặc_hết_đường, min_dist,
    mean|d|, plan_ms, hết_đường)."""
    rng = np.random.default_rng(seed)
    d0 = rng.uniform(-INIT_D_M, INIT_D_M)
    psi0 = math.radians(rng.uniform(-INIT_PSI_DEG, INIT_PSI_DEG))
    method.reset()
    ekf = FrenetEKF()
    ekf.x = np.array([d0, psi0])
    s_ego = 0.0
    v = planner_cfg.target_speed
    vision = ENV_CONFIG["vision_range_m"]
    ds_list, plan_ms = [], []
    min_dist = math.inf
    failed = False

    def dist_now():
        d = float(ekf.state.d)
        return min((math.hypot(s_o - s_ego, d_o - d) for s_o, d_o in obstacles_abs), default=math.inf)

    max_steps = int(ENV_CONFIG["max_episode_steps"] * ENV_CONFIG["dt"] / dt)
    for _ in range(max_steps):
        d, psi = float(ekf.state.d), float(ekf.state.psi)
        obstacles = [(s_o - s_ego, d_o + rng.normal(0.0, OBS_NOISE_M))
                     for s_o, d_o in obstacles_abs if 0.0 <= s_o - s_ego <= vision]
        min_dist = min(min_dist, dist_now())
        t0 = time.perf_counter()
        fp = method.plan(d, psi, v, obstacles)
        plan_ms.append((time.perf_counter() - t0) * 1000.0)
        if fp is None:
            failed = True
            break
        linear_x, angular_z, _, _ = compute_cmd_vel(fp.s, fp.d, 0.0, d, psi, v, pp_cfg)
        sim_predict(ekf, linear_x, angular_z, dt)
        s_ego += linear_x * math.cos(psi) * dt
        ds_list.append(abs(float(ekf.state.d)))
        if s_ego >= ENV_CONFIG["course_length_m"]:
            break
    min_dist = min(min_dist, dist_now())
    collided = failed or min_dist <= planner_cfg.robot_radius
    return collided, min_dist, float(np.mean(ds_list)) if ds_list else 0.0, plan_ms, failed


def evaluate(method, obstacles_abs, runs, **kw):
    coll, dists, devs, times, stuck = 0, [], [], [], 0
    for k in range(runs):
        c, md, dev, pt, f = run_once(method, obstacles_abs, seed=1000 + k, **kw)
        coll += c; stuck += f; dists.append(md); devs.append(dev); times += pt
    return dict(coll=coll, stuck=stuck, dists=dists, devs=devs, times=times)


def fmt_row(name, label, r, runs, has_obs):
    dist_s = "—" if not has_obs else f"{np.mean(r['dists']):.2f} ± {np.std(r['dists']):.2f}"
    note = f" (hết đường {r['stuck']})" if r["stuck"] else ""
    return (f"{name:20s} {label:10s} {r['coll']:>4d}/{runs:<3d} "
            f"{np.mean(r['times']):8.2f} ({np.percentile(r['times'], 95):5.2f}) "
            f"{dist_s:>15s} {np.mean(r['devs']):7.3f} ± {np.std(r['devs']):.3f}{note}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="?", default=os.path.join(MODELS_DIR, "sac_frenet_straight_multi_s1_200k"))
    ap.add_argument("--runs", type=int, default=20)
    ap.add_argument("--rate", type=float, default=ENV_CONFIG["pp_rate_hz"],
                    help="tần số pure pursuit + lập đường [Hz]")
    ap.add_argument("--multi", action="store_true", help="kịch bản nhiều vật cản (Frenet lọc trước)")
    args = ap.parse_args()
    dt = 1.0 / args.rate
    planner_cfg, pp_cfg = PLANNER_CFG, PP_CFG

    rl, frenet = RLPlanner(args.model, planner_cfg), FrenetPlanner(planner_cfg)
    kw = dict(pp_cfg=pp_cfg, dt=dt, planner_cfg=planner_cfg)
    header = (f"{'trường hợp':20s} {'phương pháp':10s} {'va chạm':>8s} {'plan_ms TB (p95)':>18s} "
              f"{'min_dist [m]':>15s} {'mean|d| [m]':>15s}")
    print(f"model: {args.model} | {args.runs} lượt/trường hợp, nhiễu d_obs σ={OBS_NOISE_M} m"
          f" | tham số robot, lookahead={pp_cfg.lookahead_distance} m, {args.rate:.0f} Hz\n")

    if not args.multi:
        print(header); print("-" * 92)
        for name, obs in SCENARIOS:
            for label, method in (("RL (SAC)", rl), ("Frenet", frenet)):
                print(fmt_row(name, label, evaluate(method, obs, args.runs, **kw), args.runs, bool(obs)), flush=True)
            print()
        return

    # Pha 1: Frenet trên mọi kịch bản -> lọc kịch bản khả thi.
    print("Pha 1 — Frenet chạy trước:")
    print(header); print("-" * 92)
    fr_res = {}
    for name, obs in MULTI_SCENARIOS:
        fr_res[name] = evaluate(frenet, obs, args.runs, **kw)
        print(fmt_row(name, "Frenet", fr_res[name], args.runs, True), flush=True)
    ok = [(n, o) for n, o in MULTI_SCENARIOS if fr_res[n]["coll"] == 0]
    bad = [(n, o) for n, o in MULTI_SCENARIOS if fr_res[n]["coll"] > 0]
    print(f"\nFrenet qua được {len(ok)}/{len(MULTI_SCENARIOS)} kịch bản.\n")

    print("Pha 2 — so sánh trên kịch bản Frenet qua được:")
    print(header); print("-" * 92)
    for name, obs in ok:
        print(fmt_row(name, "RL (SAC)", evaluate(rl, obs, args.runs, **kw), args.runs, True), flush=True)
        print(fmt_row(name, "Frenet", fr_res[name], args.runs, True))
        print()
    if bad:
        print("Kịch bản Frenet KHÔNG qua (tham khảo, không tính vào so sánh):")
        print(header); print("-" * 92)
        for name, obs in bad:
            print(fmt_row(name, "RL (SAC)", evaluate(rl, obs, args.runs, **kw), args.runs, True), flush=True)
            print(fmt_row(name, "Frenet", fr_res[name], args.runs, True))
            print()


if __name__ == "__main__":
    main()
