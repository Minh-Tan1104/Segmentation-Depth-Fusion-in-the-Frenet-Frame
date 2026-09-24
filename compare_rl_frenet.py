"""So sánh RL (SAC) với Frenet cost-based trên 4 trường hợp: vật cản trái,
phải, giữa, không vật cản. 3 chỉ số:

- plan_ms:  thời gian chọn quỹ đạo mỗi tick (RL: RLPolicy + gate + build +
            check như PlannerLogic._plan_rl; Frenet: FrenetOptimalPlanner.plan).
- min_dist: khoảng cách nhỏ nhất từ XE (vị trí thật) tới vật cản [m].
- mean|d|:  độ lệch ngang trung bình so với tâm làn cả lượt chạy [m].

Hai phương pháp chạy CÙNG vòng mô phỏng (compute_cmd_vel + sim_predict của
train_frenet_rl.py) và cùng seed từng lượt: nhiễu đo
d_obs mỗi tick + lệch nhẹ điều kiện đầu. RL dùng action liên tục như trên xe.

    python3 compare_rl_frenet.py [model] [--runs N] [--rate HZ]

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
from planner_motion.rl_policy import RLPolicy
from train_frenet_rl import ENV_CONFIG, PLANNER_CFG, PP_CFG, MODELS_DIR, sim_predict

import os

S_OBS = 30.0
SCENARIOS = [
    ("vật cản trái", -0.5),
    ("vật cản phải", +0.5),
    ("vật cản giữa", 0.0),
    ("không vật cản", None),
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
            if fp is not None and self.planner._obstacle_cost(fp, obstacles) == 0.0:
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


def run_once(method, d_obs, seed, pp_cfg=PP_CFG, dt=1.0 / ENV_CONFIG["pp_rate_hz"], planner_cfg=PLANNER_CFG):
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
    max_steps = int(ENV_CONFIG["max_episode_steps"] * ENV_CONFIG["dt"] / dt)
    for _ in range(max_steps):
        d, psi = float(ekf.state.d), float(ekf.state.psi)
        obstacles = []
        if d_obs is not None:
            ds = S_OBS - s_ego
            if 0.0 <= ds <= vision:
                obstacles.append((ds, d_obs + rng.normal(0.0, OBS_NOISE_M)))
            min_dist = min(min_dist, math.hypot(ds, d_obs - d))
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
    if d_obs is not None:
        min_dist = min(min_dist, math.hypot(S_OBS - s_ego, d_obs - float(ekf.state.d)))
    collided = failed or (d_obs is not None and min_dist <= planner_cfg.robot_radius)
    return collided, min_dist, float(np.mean(ds_list)), plan_ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="?", default=os.path.join(MODELS_DIR, "sac_frenet_straight_robot_s1_50k"))
    ap.add_argument("--runs", type=int, default=20)
    ap.add_argument("--rate", type=float, default=ENV_CONFIG["pp_rate_hz"],
                    help="tần số pure pursuit + lập đường [Hz]")
    args = ap.parse_args()
    dt = 1.0 / args.rate
    planner_cfg, pp_cfg = PLANNER_CFG, PP_CFG

    methods = [("RL (SAC)", RLPlanner(args.model, planner_cfg)), ("Frenet", FrenetPlanner(planner_cfg))]
    print(f"model: {args.model} | {args.runs} lượt/trường hợp, nhiễu d_obs σ={OBS_NOISE_M} m"
          f" | tham số robot, lookahead={pp_cfg.lookahead_distance} m, {args.rate:.0f} Hz\n")
    print(f"{'trường hợp':15s} {'phương pháp':10s} {'va chạm':>8s} {'plan_ms TB (p95)':>18s} "
          f"{'min_dist [m]':>15s} {'mean|d| [m]':>15s}")
    print("-" * 86)
    for name, d_obs in SCENARIOS:
        for label, method in methods:
            coll, dists, devs, times = 0, [], [], []
            for k in range(args.runs):
                c, md, dev, pt = run_once(method, d_obs, seed=1000 + k, pp_cfg=pp_cfg, dt=dt,
                                          planner_cfg=planner_cfg)
                coll += c; dists.append(md); devs.append(dev); times += pt
            dist_s = "—" if d_obs is None else f"{np.mean(dists):.2f} ± {np.std(dists):.2f}"
            print(f"{name:15s} {label:10s} {coll:>4d}/{args.runs:<3d} "
                  f"{np.mean(times):8.2f} ({np.percentile(times, 95):5.2f}) "
                  f"{dist_s:>15s} {np.mean(devs):7.3f} ± {np.std(devs):.3f}")
        print()


if __name__ == "__main__":
    main()
