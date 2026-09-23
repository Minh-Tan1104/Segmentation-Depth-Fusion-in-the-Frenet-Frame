"""Train SAC agent thay cho hàm cost (chọn lateral offset di) của Frenet
planner straight-mode (planner_motion/frenet_planner.py).

Phạm vi RL action = 2 chiều: [d_target, Ti] — d_target TUYỆT ĐỐI (map tuyến
tính action[0]∈[-1,1] -> [D_MIN_OFFSET, D_MAX_OFFSET], xem _decode_action),
KHÔNG phải Δd_target (thiết kế ban đầu, đã đổi sau khi quan sát training
thật: Δd_target tích luỹ qua nhiều step dễ trôi dần ra biên d_max mà không
có cơ chế tự kéo về 0 khi hết obstacle). `tv` (tốc độ đích dọc) GIỮ CỐ ĐỊNH
= target_speed — quyết định đã thống nhất, để RL chỉ thay đúng phần "chọn
quỹ đạo/lateral offset" mà đề bài yêu cầu, không đụng vào phần chọn tốc độ
dọc.

Tái dùng NGUYÊN VẸN, không viết lại công thức:
  - QuinticPolynomial/QuarticPolynomial, FrenetOptimalPlanner._calc_global_paths,
    FrenetOptimalPlanner._check_paths (planner_motion/frenet_planner.py)
    -> sinh quỹ đạo + check hard constraint (tốc độ/gia tốc/độ cong/va chạm)
       giống hệt planner gốc.
  - compute_cmd_vel (control/pure_pursuit.py) -> Pure Pursuit y hệt control_node.
  - FrenetEKF.predict (control/ekf.py) -> kinematics cập nhật d/psi y hệt EKF
    thật (cùng quy ước dấu +d=phải reference, psi panel = -psi_ROS).

Không sửa curve mode / không sửa các file trên.
"""

from __future__ import annotations

import math
import os

import numpy as np

import matplotlib
# Hiện cửa sổ trực tiếp lúc train (TkAgg) nếu máy có display (tkinter khả
# dụng VÀ có $DISPLAY thật) — chỉ import tkinter thành công KHÔNG đủ, tạo Tk
# window khi không có $DISPLAY (server/CI/background job không X) crash bằng
# TclError giữa lúc train, nên phải check cả 2. Fallback về "Agg" (headless,
# chỉ lưu file) khi không đủ điều kiện, để script không bao giờ crash vì lý
# do hiển thị.
try:
    import tkinter  # noqa: F401
    if not os.environ.get("DISPLAY"):
        raise RuntimeError("no $DISPLAY")
    matplotlib.use("TkAgg")
    INTERACTIVE_PLOTS = True
except Exception:
    matplotlib.use("Agg")
    INTERACTIVE_PLOTS = False
import matplotlib.pyplot as plt
if INTERACTIVE_PLOTS:
    plt.ion()

import gymnasium as gym
from gymnasium import spaces

from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

import cv2

from control.ekf import FrenetEKF
from control.pure_pursuit import PurePursuitConfig, compute_cmd_vel
from planner_motion.frenet_planner import (
    FrenetOptimalPlanner,
    FrenetPlannerConfig,
    FrenetPath,
)
from planner_motion.rl_policy import (
    RLPolicyMeta,
    build_observation,
    decode_action,
    meta_path,
    obs_high,
)
from visualization.logic import OverlayRenderer


# ─────────────────────────────────────────────────────────────────────────
# Cấu hình planner — khớp ĐÚNG default ROS param của planner_motion/node.py
# (Declare các "plan_*" param) để RL và cost-based planner so sánh công bằng
# trên cùng 1 bộ giới hạn/candidate range. Field không expose qua ROS param
# (k_j, k_t, k_lat, k_lon, dt, d_t_s, n_s_sample, min_path_*) giữ nguyên
# default của FrenetPlannerConfig, vì deployment thật cũng dùng default đó.
# ─────────────────────────────────────────────────────────────────────────
PLANNER_CFG = FrenetPlannerConfig(
    target_speed=2.0,          # plan_speed
    max_speed=2.0 * 1.5,       # planner_motion/logic.py: max_speed=plan_speed*1.5
    robot_radius=0.6,          # plan_robot_radius
    max_road_width=2.5,        # plan_road_width
    d_road_w=0.4,              # plan_d_road_w
    max_curvature=1.5,         # plan_max_curvature
    clearance=1.2,             # plan_clearance
    k_obs=10.0,                # plan_obstacle_weight
    center_offset=0.0,         # plan_center_offset
    k_d=1.0,                   # plan_center_weight
    min_t=3.5,                 # plan_min_horizon_s
    max_t=4.0,                 # plan_max_horizon_s
)
# Khoảng lateral offset ứng viên mà cost-based planner đang dùng (xem
# _calc_frenet_paths: n_steps=round(max_road_width/d_road_w), di =
# center_offset ± n_steps*d_road_w) — RL action map vào ĐÚNG khoảng này để
# so sánh công bằng.
_N_STEPS = max(1, round(PLANNER_CFG.max_road_width / PLANNER_CFG.d_road_w))
D_MAX_OFFSET = PLANNER_CFG.center_offset + _N_STEPS * PLANNER_CFG.d_road_w  # = 2.4
D_MIN_OFFSET = PLANNER_CFG.center_offset - _N_STEPS * PLANNER_CFG.d_road_w  # = -2.4

PP_CFG = PurePursuitConfig()  # default dataclass — chưa có override ROS param cụ thể

# Tầm nhìn obstacle của observation (POMDP: vật lý luôn "có thật" trong env kể
# cả ngoài tầm nhìn, chỉ observation bị giới hạn). Khớp ĐÚNG khoảng xa nhất mà
# 1 quỹ đạo/tick THỰC SỰ vươn tới được (Ti*target_speed, Ti∈[min_t,max_t]=
# [3.5,4.0]s, target_speed=2m/s -> path dài 7-8m) — trước đây để 15.0 (xa hơn
# nhiều so với path 1 tick có thể chạm tới) khiến obstacle "thấy được" rất
# lâu trước khi path nào có thể phản ứng, tạo nhiễu dư khiến policy học lệch
# tâm nhẹ ngay cả ở đoạn chưa cần né (xem lịch sử debug: d trôi tới +0.53m
# lúc obstacle CÒN NGOÀI TẦM NHÌN CŨ 15m, tức không phải do rò obstacle — mà
# do noise huấn luyện tích luỹ từ việc thấy obstacle quá sớm/quá lâu).
VISION_RANGE_M = PLANNER_CFG.max_t * PLANNER_CFG.target_speed  # = 8.0

# Hằng số mã hoá observation/action — lưu cạnh model (.meta.json) để lúc chạy
# thật (control_node, plan_use_rl=true) giải mã y hệt lúc train.
RL_META = RLPolicyMeta(
    center_offset=PLANNER_CFG.center_offset,
    d_min=D_MIN_OFFSET,
    d_max=D_MAX_OFFSET,
    min_t=PLANNER_CFG.min_t,
    max_t=PLANNER_CFG.max_t,
    vision_range_m=VISION_RANGE_M,
)

MODELS_DIR = "models"
PLOTS_DIR = "plots"
TB_LOG_DIR = "tb_logs"

# Panel per-tick (VizCallback) — kích thước canvas cv2, giống panel
# "Frenet/EKF" của visualization/logic.py:draw_frenet_panel. lane_width_m
# chỉ để VẼ (đường lane đứt nét ±lane_width_m) — không phải giới hạn
# planner, chọn = D_MAX_OFFSET để khớp trực quan với khoảng d_target.
PANEL_W, PANEL_H = 380, 460
PANEL_LANE_WIDTH_M = D_MAX_OFFSET


# ─────────────────────────────────────────────────────────────────────────
# Cấu hình môi trường + reward.
#
# Reward CHÍNH LÀ cost của Frenet Optimal Trajectory gốc, tính TRÊN ĐÚNG path
# mà RL vừa sinh mỗi step — không phải reward tự thiết kế riêng. Mỗi step:
#     cost = fp.cf + k_obs * obstacle_cost(fp, obstacles)
#     reward = -cost
# (fp.cf = k_lat*cd + k_lon*cv, cd/cv tính bằng FrenetOptimalPlanner._path_cost
# — TÁCH ra từ planner_motion/frenet_planner.py:_calc_frenet_paths, đúng công
# thức mà cost-based planner dùng để argmin chọn di; obstacle_cost = đúng
# FrenetOptimalPlanner._obstacle_cost). Cost-based planner CHỌN di/Ti/tv để
# minimize đúng lượng này trong 1 tập rời rạc; RL ở đây học minimize CÙNG
# lượng đó qua policy gradient trên action liên tục — nên hành vi hội tụ kỳ
# vọng giống nhau (jerk thấp, về center, né vật an toàn), không cần thêm
# w_d/w_psi/w_smooth/w_safety tự chế nữa.
#
# r_collision/r_offlane/r_goal là 3 tín hiệu KHÔNG có trong cost per-tick gốc
# (cost-based planner không có khái niệm "episode" — nó lọc path không khả
# thi bằng _check_paths rồi loại hẳn khỏi tập argmin, không "phạt" gì cả).
# RL cần tín hiệu terminal tương đương vì phải học qua nhiều step liên tiếp.
# ─────────────────────────────────────────────────────────────────────────
ENV_CONFIG = {
    # bước mô phỏng RL: gộp 1 chu kỳ "replan + pure pursuit + di chuyển" của
    # control_node thật (planner ~15Hz, control ~50Hz) thành 1 step duy nhất
    # để train nhanh hơn — không ảnh hưởng tính đúng của kinematics/Frenet
    # convention vì vẫn gọi đúng compute_cmd_vel + FrenetEKF.predict mỗi step.
    "dt": 0.5,
    "course_length_m": 60.0,        # chiều dài đoạn đường mỗi episode
    "max_episode_steps": 200,       # an toàn: tránh episode chạy vô hạn
    "vision_range_m": VISION_RANGE_M,
    "d_max_offset_m": D_MAX_OFFSET,     # = 2.4 — cũng dùng làm ngưỡng "lệch khỏi làn"

    "r_collision": 100.0,   # phạt lớn khi path không khả thi (va chạm HOẶC
                             # vi phạm tốc độ/gia tốc/độ cong — xem _check_paths)
    "r_offlane": 50.0,      # phạt khi |d| vượt d_max_offset_m (kết thúc episode)
    # r_goal=200 (không phải 50): đo thật cho thấy cost/step trung bình khi
    # lái tốt ~0.7-2.3 (chủ yếu do k_t*Ti — "phí thời gian" luôn cộng dồn dù
    # không lệch tâm/không obstacle); course_length=60m ở target_speed=2m/s
    # mất ~60 step để đi hết -> 1 lượt hoàn thành TỐT vẫn tốn tổng cost ~-45
    # đến -140. r_goal=50 chỉ bù được 1 phần nhỏ, khiến "đi hết đường an
    # toàn" không rõ ràng tốt hơn "bỏ cuộc sớm bằng off-lane" (~-50 đến -80,
    # episode ngắn nên cộng dồn ít) — 2 lượt train thật (xem lịch sử debug)
    # đều cho thấy agent học rush ra d_max thay vì né đúng cách. Nâng lên
    # 200 để hoàn thành course rõ ràng là lựa chọn tốt nhất, áp đảo mọi
    # chiến lược "thoát sớm".
    "r_goal": 200.0,
    # Tỉ lệ episode có obstacle đặt SÁT GIỮA làn (|d_obs| <= band). Đo thật:
    # policy chọn phía né theo DẤU của d_obs, nên tại d_obs≈0 (không có dấu)
    # nó đi thẳng vào obstacle — va chạm 70-85% tại đúng d_obs=0, còn lệch
    # 0.1m là né tốt 100%. Với d_obs random đều ±1.8m, ca này chỉ ~2% episode
    # nên gần như không được học. Tăng mẫu để nhiễu explore tìm ra 1 phía
    # né thành công và policy học theo.
    "center_obstacle_prob": 0.2,
    "center_obstacle_band_m": 0.1,
}

# use_sde: bật vì action là target liên tục cần khám phá "mượt theo thời
# gian" (state-dependent exploration giữ nhiễu tương quan qua các step, hợp
# với việc dò d_target/Ti thay vì nhiễu độc lập từng step như mặc định) —
# đổi False nếu muốn quay lại nhiễu Gaussian độc lập chuẩn.
SAC_KWARGS = dict(
    policy="MlpPolicy",
    learning_rate=3e-4,
    buffer_size=200_000,
    batch_size=256,
    train_freq=1,
    gradient_steps=1,
    # SB3 default learning_starts=100 quá thấp cho env này: đo thật (train
    # 80k step) cho thấy policy bắt đầu gradient update gần như ngay lập tức
    # trên rất ít dữ liệu ngẫu nhiên, hội tụ sớm vào 1 heuristic thô ("luôn
    # đánh lái ra hết biên d_max mỗi khi thấy obstacle", bất kể obstacle ở xa
    # hay ở phía nào) — off_lane_rate leo từ ~20% lên ~86% trong ~300 episode
    # đầu trong khi reward chững/xấu đi, ent_coef giảm nhanh (~1.0 -> 0.098)
    # khoá luôn heuristic đó lại trước khi có đủ dữ liệu để sửa. Tăng
    # learning_starts để thu thập nhiều transition ngẫu nhiên đa dạng hơn
    # (nhiều tình huống obstacle trái/phải/không có) trước khi bắt đầu học.
    learning_starts=5000,
    use_sde=True,
    sde_sample_freq=8,
    verbose=1,
)


def _round_to_grid(d_value: float, d_road_w: float, center_offset: float) -> float:
    """Làm tròn d_value về lưới candidate rời rạc mà cost-based planner đang
    dùng (center_offset + k*d_road_w) — dùng khi so sánh công bằng SAC vs
    cost-based (xem test_frenet_rl.py)."""
    steps = round((d_value - center_offset) / d_road_w)
    return center_offset + steps * d_road_w


class FrenetStraightEnv(gym.Env):
    """Env RL straight-mode: reference thẳng (x=s, y=d, giống hệt quy ước
    trong planner_motion/frenet_planner.py), obstacle tĩnh random mỗi
    episode. Action chọn (d_target, Ti); quỹ đạo sinh bằng ĐÚNG
    QuinticPolynomial/QuarticPolynomial + check hard-constraint của project."""

    metadata = {"render_modes": []}

    def __init__(self, config: dict | None = None, discretize_d_for_eval: bool = False):
        super().__init__()
        self.cfg = {**ENV_CONFIG, **(config or {})}
        self.planner_cfg = PLANNER_CFG
        self._planner = FrenetOptimalPlanner(self.planner_cfg)
        # True khi dùng để so sánh công bằng với cost-based planner (test
        # script) — làm tròn d_target về lưới rời rạc trước khi sinh path,
        # vì cost-based planner chỉ bao giờ chọn được giá trị trên lưới đó.
        self.discretize_d_for_eval = discretize_d_for_eval

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        # obs = [d, psi, ds_obs (đã scale về ~[0, D_MAX_OFFSET]), d_obs] — mã
        # hoá/biên dùng chung với lúc chạy thật (planner_motion/rl_policy.py).
        high = obs_high(RL_META)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

        self.ekf = FrenetEKF()
        self.s_ego = 0.0
        self.obstacle: tuple[float, float] | None = None
        self.step_count = 0
        self._rng = np.random.default_rng()

    # ── Gymnasium API ───────────────────────────────────────────────────
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        options = options or {}

        d0 = options.get("d0")
        psi0 = options.get("psi0")
        if d0 is None:
            d0 = float(self._rng.uniform(-0.3, 0.3))
        if psi0 is None:
            psi0 = float(self._rng.uniform(math.radians(-5.0), math.radians(5.0)))
        self.ekf = FrenetEKF()
        self.ekf.x = np.array([d0, psi0])

        if "obstacle" in options:
            self.obstacle = options["obstacle"]
        else:
            has_obstacle = self._rng.uniform() > 0.3
            if has_obstacle:
                s_obs = float(self._rng.uniform(10.0, self.cfg["course_length_m"] - 5.0))
                if self._rng.uniform() < self.cfg["center_obstacle_prob"]:
                    band = self.cfg["center_obstacle_band_m"]
                    d_obs = float(self._rng.uniform(-band, band))
                else:
                    d_obs = float(self._rng.uniform(D_MIN_OFFSET * 0.75, D_MAX_OFFSET * 0.75))
                self.obstacle = (s_obs, d_obs)
            else:
                self.obstacle = None

        self.s_ego = 0.0
        self.step_count = 0

        return self._obs(), {}

    def step(self, action: np.ndarray):
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        cfg = self.cfg
        d_target, Ti = self._decode_action(action)

        psi_before = float(self.ekf.state.psi)
        d_before = float(self.ekf.state.d)
        c_speed = self.planner_cfg.target_speed
        c_d_d = c_speed * math.sin(psi_before)

        fp = self._build_path(d_before, c_d_d, d_target, Ti, c_speed)

        obstacles_relative = []
        if self.obstacle is not None:
            s_obs_abs, d_obs = self.obstacle
            obstacles_relative.append((s_obs_abs - self.s_ego, d_obs))

        # Reuse NGUYÊN VẸN check hard-constraint (va chạm/tốc độ/gia
        # tốc/độ cong) của planner gốc — path không lọt qua đây coi như
        # "infeasible", y hệt lý do cost-based planner loại nó ở _check_paths.
        feasible = len(self._planner._check_paths([fp], obstacles_relative)) > 0
        closest_dist = self._closest_obstacle_dist(fp, obstacles_relative)
        # cost = ĐÚNG lượng cost-based planner minimize (plan(): cost = fp.cf +
        # k_obs*obstacle_cost) — reuse nguyên _obstacle_cost, không viết lại.
        cost = fp.cf + self.planner_cfg.k_obs * self._planner._obstacle_cost(fp, obstacles_relative)

        linear_x, angular_z, target_s, target_d = compute_cmd_vel(
            fp.s, fp.d, 0.0, d_before, psi_before, c_speed, PP_CFG,
        )
        # Kinematics d/psi: reuse NGUYÊN VẸN FrenetEKF.predict (cùng quy ước
        # dấu +d=phải reference, psi panel=-psi_ROS như control/ekf.py).
        #
        # v_odom TRUYỀN ÂM (-linear_x), CHỈ ở đây (predict), không đụng file
        # control/ekf.py hay control/pure_pursuit.py: 2 file đó dùng 2 quy
        # ước d_dot NGƯỢC NHAU (đã verify bằng thực nghiệm — xem lịch sử
        # debug) — predict() tích phân d_dot=-v*sin(psi), còn compute_cmd_vel
        # (heading_gain*psi_now, comment "d_dot=v*sin(psi)") được thiết kế
        # theo d_dot=+v*sin(psi) (khớp planner_motion/logic.py: c_d_d =
        # target_speed*sin(hdg), KHÔNG có dấu trừ). Gọi predict() nguyên vẹn
        # với v_odom=-linear_x cho ra đúng d_dot=+v*sin(psi) mà pure_pursuit
        # cần để vòng lặp hội tụ (đã verify: dùng đúng dấu gốc thì d/psi
        # PHÂN KỲ ngay cả với lệch heading nhỏ 5°; đảo dấu ở đây thì hội tụ
        # ổn định). CHỈ áp dụng trong phạm vi mô phỏng RL — không sửa 2 file
        # gốc vì chúng ảnh hưởng robot thật, chưa rõ đây là bug hay có ngữ
        # cảnh bù trừ khác ở hệ thật (theo yêu cầu người dùng).
        self.ekf.predict(v_odom=-linear_x, omega_odom=angular_z, dt=cfg["dt"])
        delta_s = linear_x * math.cos(psi_before) * cfg["dt"]
        self.s_ego += delta_s

        d_after = float(self.ekf.state.d)
        psi_after = float(self.ekf.state.psi)

        reward, terminated, info = self._reward_and_done(
            d_after, psi_after, feasible, cost, closest_dist
        )
        self.step_count += 1
        truncated = self.step_count >= cfg["max_episode_steps"]

        # Dữ liệu path/pose CỦA TICK NÀY (trước khi update sang tick sau) —
        # không ảnh hưởng reward/kinematics, chỉ để VizCallback vẽ panel
        # per-tick kiểu draw_frenet_panel (visualization/logic.py).
        info["path_d"] = fp.d.tolist()
        info["path_s"] = fp.s.tolist()
        info["d_before"] = d_before
        info["psi_before"] = psi_before
        info["s_ego_before"] = self.s_ego - delta_s
        info["d_target"] = d_target
        info["Ti"] = Ti
        info["target_s"] = target_s
        info["target_d"] = target_d
        info["obstacle"] = self.obstacle
        # Lệnh Pure Pursuit THẬT đã áp dụng cho tick này — chỉ để caller (test
        # script) phát lại chuyển động mượt cho hiển thị real-time (xem
        # test_frenet_rl.py:_animate_substeps), không ảnh hưởng state/reward.
        info["linear_x"] = linear_x
        info["angular_z"] = angular_z

        return self._obs(), reward, terminated, truncated, info

    # ── nội bộ ───────────────────────────────────────────────────────────
    def _decode_action(self, action: np.ndarray) -> tuple[float, float]:
        cfg = self.cfg
        # d_target TUYỆT ĐỐI (không phải Δd_target) — map tuyến tính
        # action[0]∈[-1,1] -> [D_MIN_OFFSET, D_MAX_OFFSET]. ĐỔI từ thiết kế
        # Δd_target ban đầu (xem lịch sử: lý do lúc đó là tránh dao động
        # ping-pong) sau khi quan sát training THẬT: Δd_target tích luỹ qua
        # prev_d_target không có cơ chế "kéo về 0" khi không có obstacle —
        # policy dễ trôi dần ra biên d_max dù input hiện tại KHÔNG có obstacle
        # (xác nhận bằng panel thật: agent lái gần hết biên khi khung hình
        # không có obstacle nào) -> off_lane_rate mắc kẹt ~85-92% suốt nhiều
        # trăm episode dù collision đã học tốt. d_target tuyệt đối là hàm
        # TRỰC TIẾP của state hiện tại (d, psi, d_obs, ds_obs) mỗi step,
        # không phụ thuộc lịch sử -> khi obs "không obstacle" (giá trị mặc
        # định cố định), policy chỉ cần học 1 ánh xạ tĩnh về d_target≈0,
        # không cần "nhớ" đã trôi bao xa để tự kéo lại.
        d_target, Ti = decode_action(action, RL_META)
        if self.discretize_d_for_eval:
            d_target = _round_to_grid(d_target, self.planner_cfg.d_road_w, self.planner_cfg.center_offset)
            d_target = float(np.clip(d_target, D_MIN_OFFSET, D_MAX_OFFSET))
        return d_target, Ti

    def _build_path(self, c_d: float, c_d_d: float, d_target: float, Ti: float, tv: float) -> FrenetPath:
        """Sinh 1 quỹ đạo DUY NHẤT bằng đúng lớp polynomial + cách chuyển
        Cartesian mà _calc_frenet_paths/_calc_global_paths dùng (planner_motion/
        frenet_planner.py) — chỉ khác: không enumerate nhiều (di, Ti, tv), vì
        (d_target, Ti) đã do RL chọn thẳng, tv giữ cố định = target_speed.
        fp.cf = ĐÚNG cost cost-based planner tính (build_path gọi _path_cost)
        — reward dùng thẳng fp.cf. Cùng hàm với nhánh RL lúc chạy thật
        (planner_motion/logic.py:_plan_rl)."""
        fp = self._planner.build_path(0.0, tv, c_d, c_d_d, 0.0, d_target, Ti, tv)
        FrenetOptimalPlanner._calc_global_paths([fp])
        return fp

    @staticmethod
    def _closest_obstacle_dist(fp: FrenetPath, obstacles: list[tuple[float, float]]) -> float:
        if not obstacles or len(fp.x) == 0:
            return math.inf
        best = math.inf
        for ox, oy in obstacles:
            d = float(np.min(np.hypot(fp.x - ox, fp.y - oy)))
            best = min(best, d)
        return best

    def _obstacle_obs(self) -> tuple[float, float, float]:
        """Trả (ds_obs quan sát được, d_obs quan sát được, ds thật hoặc inf).
        Giới hạn tầm nhìn CHỈ áp dụng cho observation (POMDP) — vật lý
        (va chạm/khoảng cách an toàn) luôn dùng self.obstacle thật."""
        cfg = self.cfg
        if self.obstacle is None:
            return cfg["vision_range_m"], 0.0, math.inf
        s_obs_abs, d_obs = self.obstacle
        ds = s_obs_abs - self.s_ego
        if ds < 0.0 or ds > cfg["vision_range_m"]:
            return cfg["vision_range_m"], 0.0, math.inf
        return ds, d_obs, ds

    def _obs(self) -> np.ndarray:
        ds_obs, d_obs, ds_real = self._obstacle_obs()
        obstacle = (ds_obs, d_obs) if math.isfinite(ds_real) else None
        return build_observation(
            float(self.ekf.state.d), float(self.ekf.state.psi), obstacle, RL_META
        )

    def _reward_and_done(self, d, psi, feasible, cost, closest_dist):
        """reward = -cost, với cost = fp.cf + k_obs*obstacle_cost(fp, obstacles)
        — ĐÚNG lượng mà cost-based planner argmin để chọn di/Ti/tv (xem
        FrenetOptimalPlanner.plan(), planner_motion/frenet_planner.py:398-403).
        RL không tự thiết kế reward bám line/heading/mượt riêng nữa — toàn bộ
        nằm trong cf (qua _path_cost: k_j*jerk² + k_t*Ti + k_d*(d-center)²)
        và obstacle_cost (qua _obstacle_cost, cùng công thức/clearance/k_obs).
        Chỉ còn r_collision/r_offlane/r_goal là tín hiệu terminal RL cần mà
        cost-based planner (không có khái niệm episode) không có."""
        cfg = self.cfg
        _ds_obs, _d_obs, ds_real = self._obstacle_obs()

        reward = -cost

        terminated = False
        info: dict = {"collided": False, "off_lane": False, "reached_goal": False,
                      "d": d, "psi": psi, "closest_dist": closest_dist, "ds_obs_real": ds_real}

        if not feasible:
            reward -= cfg["r_collision"]
            terminated = True
            info["collided"] = True
        elif abs(d) > cfg["d_max_offset_m"]:
            reward -= cfg["r_offlane"]
            terminated = True
            info["off_lane"] = True
        elif self.s_ego >= cfg["course_length_m"]:
            reward += cfg["r_goal"]
            terminated = True
            info["reached_goal"] = True

        return reward, terminated, info


def _sweep_illustrative_candidates(
    env: "FrenetStraightEnv", d_before: float, psi_before: float, Ti: float,
) -> list[list[tuple[float, float]]]:
    """Sinh quỹ đạo ứng viên CHỈ ĐỂ MINH HOẠ panel (không dùng để chọn hành
    động — RL đã chọn (d_target, Ti) rồi) bằng cách sweep d_target qua đúng
    lưới rời rạc mà cost-based planner đang dùng (di_values, cùng d_road_w),
    giữ Ti = Ti mà RL vừa chọn ở tick này. Dùng lại NGUYÊN `_build_path`
    (Quintic/QuarticPolynomial + _calc_global_paths) — không viết công thức
    mới. Mục đích: panel vẽ "quạt xám" giống draw_frenet_panel thật, cho
    thấy các lựa chọn khác cùng thời điểm mà agent đã KHÔNG chọn."""
    cfg = env.planner_cfg
    c_speed = cfg.target_speed
    c_d_d = c_speed * math.sin(psi_before)
    n_steps = max(1, round(cfg.max_road_width / cfg.d_road_w))
    di_values = cfg.center_offset + cfg.d_road_w * np.arange(-n_steps, n_steps + 1)
    out = []
    for di in di_values:
        fp = env._build_path(d_before, c_d_d, float(di), Ti, c_speed)
        out.append(list(zip(fp.d.tolist(), fp.s.tolist())))
    return out


def render_frenet_panel(
    info: dict,
    renderer: OverlayRenderer,
    path_builder_env: "FrenetStraightEnv",
    title: str,
) -> np.ndarray | None:
    """Dựng canvas panel Frenet (draw_frenet_panel, visualization/logic.py)
    từ info CỦA 1 STEP mô phỏng (bất kể nguồn: rollout training thật,
    VizCallback, hay playback real-time lúc test — xem test_frenet_rl.py).
    Tách thành hàm module-level (không phải method riêng của VizCallback) để
    dùng chung, không viết lại cho test. Trả None nếu info thiếu field cần
    thiết (an toàn bỏ qua thay vì crash caller)."""
    required = ("path_d", "path_s", "d_before", "psi_before", "s_ego_before",
                "Ti", "target_d", "target_s")
    if any(k not in info for k in required):
        return None

    d_before = info["d_before"]
    psi_before = info["psi_before"]
    s_ego_before = info["s_ego_before"]

    candidate_paths = _sweep_illustrative_candidates(
        path_builder_env, d_before, psi_before, info["Ti"]
    )
    optimal_path = list(zip(info["path_d"], info["path_s"]))

    detections = []
    obstacle = info.get("obstacle")
    if obstacle is not None:
        s_obs_abs, d_obs = obstacle
        ds_obs = s_obs_abs - s_ego_before
        detections.append({
            "label": "obs",
            # draw_frenet_panel đọc fr_det["d_m"]/["s_m"] làm fallback
            # mặc định của .get("..._filtered", fr_det["..."]) — key đó
            # bị truy cập LUÔN (Python eval default arg trước khi gọi
            # .get) nên phải có cả 2 key dù ta chỉ có 1 giá trị.
            "frenet": {
                "available": True,
                "d_m": d_obs,
                "d_m_filtered": d_obs,
                "s_m": ds_obs,
                "s_m_filtered": ds_obs,
            },
        })

    frenet = {
        "kappa_ff": 0.0,
        "s_max_m": max(max(info["path_s"], default=0.5) * 1.15, 0.5),
        "candidate_paths": candidate_paths,
        "optimal_path": optimal_path,
        # target_s từ compute_cmd_vel đã tính với s_now=0.0 (env.step gọi
        # compute_cmd_vel(..., 0.0, ...) — s0 local convention y hệt
        # plan_from_state) nên ĐÃ tương đối theo đúng frame fp.s, KHÔNG
        # trừ s_ego_before (khác obstacle ở trên vốn lưu ở frame TUYỆT
        # ĐỐI của course nên mới cần trừ).
        "lookahead_point": [info["target_d"], info["target_s"]],
        "d_meters_filtered": -d_before,
        "heading_filtered": math.degrees(psi_before),
        "using_gps_route": False,
    }

    canvas = np.full((PANEL_H, PANEL_W, 3), 255, dtype=np.uint8)
    renderer.draw_frenet_panel(
        canvas, frenet, detections, px0=0, py0=0,
        panel_w=PANEL_W, panel_h=PANEL_H, title=title,
    )
    return canvas


# ─────────────────────────────────────────────────────────────────────────
# Visualize REAL-TIME trong lúc train
# ─────────────────────────────────────────────────────────────────────────
class VizCallback(BaseCallback):
    """Vẽ REAL-TIME bằng dữ liệu của CHÍNH rollout training (self.locals
    ["infos"] mỗi step SB3 gọi _on_step) — không chạy rollout riêng để
    visualize, nên panel cho thấy ĐÚNG hành vi agent đang explore lúc train
    (kể cả nhiễu use_sde), không phải bản "sạch" tách biệt.

    - Panel Frenet per-tick (draw_frenet_panel, y hệt UI control_node/
      visualization_node thật): cập nhật mỗi `render_every_n_steps` bước env
      -> gần như real-time, không chờ hết episode.
    - Reward curve: cập nhật mỗi khi 1 episode kết thúc.
    - Snapshot lịch sử (panel_ep<N>.png): lưu thêm 1 bản mỗi
      `snapshot_every_episodes` episode để so sánh tiến trình qua thời gian
      (panel_live.png luôn bị ghi đè, không giữ lịch sử).

    INTERACTIVE_PLOTS=True (có display): giữ 2 cửa sổ cố định, cập nhật tại
    chỗ. Không có display: chỉ ghi file (fallback Agg)."""

    def __init__(
        self,
        render_every_n_steps: int = 5,
        snapshot_every_episodes: int = 50,
        plots_dir: str = PLOTS_DIR,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.render_every_n_steps = render_every_n_steps
        self.snapshot_every_episodes = snapshot_every_episodes
        self.plots_dir = plots_dir
        self._episode_rewards: list[float] = []
        self._episode_count = 0
        self._step_count = 0
        self._renderer = OverlayRenderer(lane_width_m=PANEL_LANE_WIDTH_M)
        # Env "phụ" CHỈ để tái dùng _build_path (Quintic/QuarticPolynomial +
        # _calc_global_paths) khi sinh quạt candidate minh hoạ — không
        # rollout, state của nó không được dùng tới.
        self._path_builder_env = FrenetStraightEnv()

        self._fig_reward, self._ax_reward = plt.subplots(figsize=(6, 4))
        self._fig_panel, self._ax_panel = plt.subplots(
            figsize=(PANEL_W / 100, PANEL_H / 100)
        )
        if INTERACTIVE_PLOTS:
            self._fig_reward.show()
            self._fig_panel.show()

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        self._step_count += 1
        if infos and self._step_count % self.render_every_n_steps == 0:
            self._update_panel_live(infos[0])

        for info in infos:
            ep = info.get("episode")
            if ep is not None:
                self._episode_count += 1
                self._episode_rewards.append(float(ep["r"]))
                self._update_reward_plot()
                if self._episode_count % self.snapshot_every_episodes == 0:
                    self._save_panel_snapshot(info)
        return True

    def _update_reward_plot(self) -> None:
        ax = self._ax_reward
        ax.clear()
        rewards = np.array(self._episode_rewards)
        ax.plot(rewards, alpha=0.3, label="reward/episode")
        if len(rewards) >= 10:
            kernel = np.ones(10) / 10
            smooth = np.convolve(rewards, kernel, mode="valid")
            ax.plot(np.arange(9, len(rewards)), smooth, label="moving avg (10 ep)")
        ax.set_xlabel("episode")
        ax.set_ylabel("reward")
        ax.set_title(f"Reward theo episode (ep={self._episode_count})")
        ax.legend()
        self._fig_reward.tight_layout()
        self._fig_reward.savefig(os.path.join(self.plots_dir, "reward_curve.png"))
        if INTERACTIVE_PLOTS:
            self._fig_reward.canvas.draw_idle()
            self._fig_reward.canvas.flush_events()

    def _update_panel_live(self, info: dict) -> None:
        canvas = render_frenet_panel(
            info, self._renderer, self._path_builder_env,
            title=f"Frenet RL (train) step={self.num_timesteps}",
        )
        if canvas is None:
            return
        cv2.imwrite(os.path.join(self.plots_dir, "panel_live.png"), canvas)
        if INTERACTIVE_PLOTS:
            ax = self._ax_panel
            ax.clear()
            ax.imshow(canvas[:, :, ::-1])  # BGR (cv2) -> RGB (matplotlib)
            ax.axis("off")
            self._fig_panel.tight_layout()
            # flush_events() bơm event loop GUI (Tk) để cửa sổ thực sự vẽ
            # lại + phản hồi (kéo/resize) mà không chặn training loop
            # (non-blocking, khác plt.show()).
            self._fig_panel.canvas.draw_idle()
            self._fig_panel.canvas.flush_events()

    def _save_panel_snapshot(self, info: dict) -> None:
        canvas = render_frenet_panel(
            info, self._renderer, self._path_builder_env,
            title=f"Frenet RL (train) step={self.num_timesteps}",
        )
        if canvas is None:
            return
        fname = os.path.join(self.plots_dir, f"panel_ep{self._episode_count:05d}.png")
        cv2.imwrite(fname, canvas)


def main() -> None:
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(TB_LOG_DIR, exist_ok=True)

    raw_env = FrenetStraightEnv()
    check_env(raw_env, warn=True)

    env = Monitor(FrenetStraightEnv())
    model = SAC(env=env, tensorboard_log=TB_LOG_DIR, **SAC_KWARGS)

    callback = VizCallback(render_every_n_steps=5, snapshot_every_episodes=50, plots_dir=PLOTS_DIR)
    model.learn(total_timesteps=200_000, callback=callback, tb_log_name="sac_frenet_straight")

    model_path = os.path.join(MODELS_DIR, "sac_frenet_straight")
    model.save(model_path)
    RL_META.save(meta_path(model_path))
    print(f"Model đã lưu: {model_path}.zip (+ {meta_path(model_path)})")

    if INTERACTIVE_PLOTS:
        print("Train xong — đóng cửa sổ plot để kết thúc script.")
        plt.show(block=True)


if __name__ == "__main__":
    main()
